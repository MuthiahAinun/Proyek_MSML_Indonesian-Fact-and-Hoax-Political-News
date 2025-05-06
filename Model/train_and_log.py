import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm.auto import tqdm
from modelling import IndoBertDataset, load_and_augment_dataset, tokenize_data, evaluate_all
import mlflow
import mlflow.pytorch

# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load & augment dataset
df = load_and_augment_dataset("data/dataset_cleaned.csv")

# 3. Tokenisasi dan Dataset
encodings, labels, tokenizer = tokenize_data(df)
dataset = IndoBertDataset(encodings, labels)

# 4. Split data
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# 5. Load model
model = AutoModelForSequenceClassification.from_pretrained("cahya/distilbert-base-indonesian", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

# 6. MLflow setup
mlflow.set_experiment("HoaxDetection")

with mlflow.start_run():
    mlflow.log_param("model", "DistilBERT-Indo")
    mlflow.log_param("lr", 5e-5)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("epochs", 3)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1} — Avg Loss: {avg_loss:.4f}")

    # Save model
    model.save_pretrained("saved_model_hoax")
    tokenizer.save_pretrained("saved_model_hoax")
    mlflow.pytorch.log_model(model, "model")
    print("Model dan tokenizer berhasil disimpan.")

    # Evaluation
    val_acc = evaluate_all(model, val_loader, name="Validation")
    test_acc = evaluate_all(model, test_loader, name="Test")
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("test_accuracy", test_acc)
