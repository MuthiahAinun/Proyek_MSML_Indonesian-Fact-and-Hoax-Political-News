import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter
import mlflow
import mlflow.transformers
from mlflow.models.signature import infer_signature
import random
import re
import argparse
from huggingface_hub import HfApi, Repository
import tempfile
import subprocess
import os
from dotenv import load_dotenv
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import precision_score, recall_score, f1_score

print("✅ Script modelling.py dimulai...")

# ---------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_REPO_ID= os.getenv("HF_REPO_ID")

print("🔑 Environment variables loaded")

# ---------------------------------------------
# DATA AUGMENTATION (EDA)
# ---------------------------------------------
def random_deletion(words, p=0.1):
    if len(words) == 1:
        return words
    # hapus kata secara random dengan probabilitas p, tapi jangan sampai kosong
    new_words = [word for word in words if random.uniform(0, 1) > p]
    if len(new_words) == 0:
        # jika kosong, ambil satu kata random supaya tidak error
        new_words = [random.choice(words)]
    return new_words

def random_swap(words, n=1):
    new_words = words.copy()
    length = len(new_words)
    if length < 2:
        return new_words
    for _ in range(n):
        idx1, idx2 = random.sample(range(length), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def eda(text, num_aug=4):
    text = re.sub(r'[^\w\s]', '', text)  # hapus tanda baca
    words = text.split()
    if len(words) == 0:
        return [text] * num_aug  # jika teks kosong, return original beberapa kali
    augmented_sentences = []
    for _ in range(num_aug):
        aug_type = random.choice(['swap', 'delete'])
        if aug_type == 'swap':
            n_swap = max(1, len(words) // 10)
            new_words = random_swap(words, n=n_swap)
        else:
            new_words = random_deletion(words, p=0.1)
        augmented_sentences.append(' '.join(new_words))
    return augmented_sentences

# ---------------------------------------------
# LOAD & TOKENIZE DATA
# ---------------------------------------------
def load_and_augment_dataset(path, augment=True, num_aug=2):
    print(f"📂 Loading dataset from {path} with augment={augment}")
    df = pd.read_csv(path, compression='infer')
    df['label'] = df['label'].astype(int)
    print(f"📊 Loaded {len(df)} rows")

    if augment:
        augmented_texts = []
        augmented_labels = []
        for i, row in enumerate(df.itertuples(index=False)):
            # original text
            augmented_texts.append(row.text)
            augmented_labels.append(row.label)
            # augmented texts
            augmented_versions = eda(row.text, num_aug=num_aug)
            augmented_texts.extend(augmented_versions)
            augmented_labels.extend([row.label] * num_aug)

            if (i+1) % 1000 == 0:
                print(f"  ↳ Augmented {i+1} rows")

        df = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})
        print(f"📈 Dataset augmented to {len(df)} rows")

    return df

def tokenize_data(df):
    print("🔤 Tokenizing data...")
    assert 'text' in df.columns, "'text' column is required in dataframe"
    assert 'label' in df.columns, "'label' column is required in dataframe"
    tokenizer = AutoTokenizer.from_pretrained("cahya/distilbert-base-indonesian")
    # encodings = tokenizer(df['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt")
    texts = df['text'].tolist()
    
    batch_size = 512
    all_input_ids = []
    all_attention_mask = []

    print("Tokenizing in batches...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_encodings = tokenizer(batch_texts, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
        all_input_ids.append(batch_encodings['input_ids'])
        all_attention_mask.append(batch_encodings['attention_mask'])
        print(f"  ↳ Batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size} tokenized")

    input_ids = torch.cat(all_input_ids)
    attention_mask = torch.cat(all_attention_mask)
    encodings = {'input_ids': input_ids, 'attention_mask': attention_mask}


    labels = df['label'].tolist()
    print(f"✅ Tokenized {len(labels)} texts")
    return encodings, labels


class IndoBertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# ---------------------------------------------
def save_conf_matrix_plot(cm, labels, filename = f"confusion_matrix_{int(time.time())}.png"):
    print("📉 Menyimpan plot confusion matrix...")
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Confusion matrix saved as {filename}")
    return filename

def evaluate_all(model, loader, name="Validation"):
    print(f"⚙️ Evaluating model on {name} set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            true_labels.extend(labels.cpu().numpy().tolist())
            pred_labels.extend(preds.cpu().numpy().tolist())

    labels_set = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_set)

    # Simpan plot confusion matrix dan dapatkan path filenya
    cm_path = save_conf_matrix_plot(cm, labels_set)

    # Log artifact ke mlflow
    mlflow.log_artifact(cm_path)

    # Hitung metrik utama
    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    # Log metrik ke mlflow
    mlflow.log_metric(f"{name.lower()}_accuracy", acc)
    mlflow.log_metric(f"{name.lower()}_precision", precision)
    mlflow.log_metric(f"{name.lower()}_recall", recall)
    mlflow.log_metric(f"{name.lower()}_f1", f1)

    print(f"\n📊 {name} Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(true_labels, pred_labels, labels=labels_set))

    return acc

# ---------------------------------------------
# TRAINING
# ---------------------------------------------
def train(model, train_loader, device, epochs, lr=5e-5):
    print(f"🏋️ Mulai training selama {epochs} epochs dengan lr={lr}")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()  # pastikan model di mode training setiap epoch
        total_loss = 0
        all_labels = []
        all_preds = []

        for batch_idx, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Log loss per batch
            mlflow.log_metric("train_loss_batch", loss.item(), step=epoch * len(train_loader) + batch_idx)

            if batch_idx % 50 == 0 or batch_idx == len(train_loader):
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)

        print(f"✅ Epoch {epoch+1}/{epochs} selesai, Rata-rata Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}")

        # Log metrik epoch
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("train_precision", precision, step=epoch)
        mlflow.log_metric("train_recall", recall, step=epoch)
        mlflow.log_metric("train_f1", f1, step=epoch)
        mlflow.log_metric("train_accuracy", accuracy, step=epoch)

# ---------------------------------------------
# PUSH HELPERS
# ---------------------------------------------
def push_to_hf(model_dir, repo_id, hf_token):
    api = HfApi()
    print(f"⬆️ Uploading files from {model_dir} to repo {repo_id}")

    if not os.path.isdir(model_dir):
        raise ValueError(f"❌ Directory not found: {model_dir}")

    temp_dir = tempfile.mkdtemp()
    # Salin semua isi model_dir ke temp_dir
    shutil.copytree(model_dir, temp_dir, dirs_exist_ok=True)

    # Upload dari temp_dir, bukan model_dir
    for root, _, files in os.walk(temp_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            path_in_repo = os.path.relpath(file_path, temp_dir)
            print(f"Uploading {file_path} as {path_in_repo} ...")

            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=hf_token,
                repo_type="model",
                commit_message=f"Update {path_in_repo} from training"
            )
    print("✅ Upload selesai ke Hugging Face Hub")

# ---------------------------------------------
# MAIN
# ---------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return v.lower() in ("yes", "true", "t", "1")

def main():
    print("🚀 Starting main process...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data-path', type=str, default='dataset_cleaned_prepo.gz')
    parser.add_argument("--augment", type=str2bool, default=True)
    parser.add_argument("--dagshub-token", type=str, default=os.environ.get("DAGSHUB_TOKEN"))
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--mlflow-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-user", type=str, default=os.environ.get("MLFLOW_TRACKING_USERNAME"))
    parser.add_argument("--mlflow-pass", type=str, default=os.environ.get("MLFLOW_TRACKING_PASSWORD"))
    parser.add_argument("--github-repo", type=str, default=os.environ.get("GITHUB_REPO"))
    parser.add_argument("--github-token", type=str, default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--hf-repo-id", type=str, default=os.environ.get("HF_REPO_ID"))

    args = parser.parse_args()

    print(f"⚙️ Arguments: {args}")

    print("📡 Setting MLflow tracking URI...")
    mlflow.set_tracking_uri(args.mlflow_uri)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device yang digunakan: {device}")

    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run():
        # Log parameter umum
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", 5e-5)

        # Load dataset dan augmentasi
        df = load_and_augment_dataset(args.data_path, augment=args.augment)
        encodings, labels = tokenize_data(df)
        dataset = IndoBertDataset(encodings, labels)

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        print(f"📊 Dataset dibagi menjadi train {train_size} , val {val_size}, dan test {test_size} ")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Load pretrained model
        print("📥 Memuat pretrained model...")
        model = AutoModelForSequenceClassification.from_pretrained("cahya/distilbert-base-indonesian", num_labels=2)

        # Train model
        train(model, train_loader, device, epochs=args.epochs)

        # Evaluate
        val_acc = evaluate_all(model, val_loader)
        mlflow.log_metric("val_accuracy", val_acc)

        # Save model ke folder lokal untuk di-commit dan push
        model_dir = "./model_saved"
        print(f"💾 Menyimpan model ke {model_dir}")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        model.save_pretrained(model_dir)

        # Simpan tokenizer juga
        tokenizer = AutoTokenizer.from_pretrained("cahya/distilbert-base-indonesian")
        tokenizer.save_pretrained(model_dir)

        # Log model ke MLflow lengkap dengan signature dan contoh input
        print("📦 Log model ke MLflow...")
        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=0 if torch.cuda.is_available() else -1)

        # Log model ke MLflow lengkap dengan signature dan input_example
        print("📦 Log model ke MLflow...")
        input_example = {"text": "Ini contoh teks berita hoaks"}
        output_example = pipeline(input_example)
        signature = infer_signature(input_example, output_example)

        mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="model",
            input_example=input_example,
            task="text-classification",
            signature=signature,
            device=0 if torch.cuda.is_available() else -1
        )


        # Push ke Hugging Face Hub
        if args.hf_token and args.hf_repo_id:
            push_to_hf(model_dir, args.hf_repo_id, args.hf_token)


        print("🎉 Selesai seluruh proses training, evaluasi, dan push model")

if __name__ == "__main__":
    main()
