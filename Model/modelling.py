# smsml_hoax_detection_project/Model/modelling.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import random
import re

# Easy Data Augmentation (EDA) functions
def random_deletion(words, p=0.1):
    if len(words) == 1:
        return words
    return [word for word in words if random.uniform(0, 1) > p]

def random_swap(words, n=1):
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def eda(text, num_aug=4):
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    augmented_sentences = []
    for _ in range(num_aug):
        aug_type = random.choice(['swap', 'delete'])
        if aug_type == 'swap':
            new_words = random_swap(words, n=max(1, len(words) // 10))
        else:
            new_words = random_deletion(words, p=0.1)
        augmented_sentences.append(' '.join(new_words))
    return augmented_sentences

# Load and augment data
df = pd.read_csv("Experiment/preprocessing/dataset_cleaned.gz")
df = df.dropna(subset=["clean_text", "hoax"])
df['hoax'] = df['hoax'].astype(int)

# Oversample minority class
hoax_texts = df[df['hoax'] == 1]['clean_text'].tolist()
augmented_texts = []
for text in hoax_texts:
    augmented_texts.extend(eda(text, num_aug=2))

temp_df = pd.DataFrame({'clean_text': augmented_texts, 'hoax': 1})
df = pd.concat([df, temp_df], ignore_index=True)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("cahya/distilbert-base-indonesian")
encodings = tokenizer(df['clean_text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
labels = df['hoax'].tolist()

# Dataset class
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

dataset = IndoBertDataset(encodings, labels)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Training
model = AutoModelForSequenceClassification.from_pretrained("cahya/distilbert-base-indonesian", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

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

    print(f"Epoch {epoch+1} - Avg Loss: {total_loss / len(train_loader):.4f}")

# Save model
model.save_pretrained("saved_model_hoax")
tokenizer.save_pretrained("saved_model_hoax")
print("Model dan tokenizer berhasil disimpan.")

# Evaluation function
def evaluate_all(model, loader, name="Validation"):
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

    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(true_labels, pred_labels, target_names=["non-hoax", "hoax"]))
    print("Distribusi prediksi:", Counter(pred_labels))
    print("Distribusi label asli:", Counter(true_labels))

evaluate_all(model, val_loader, name="Validation")
evaluate_all(model, test_loader, name="Test")
