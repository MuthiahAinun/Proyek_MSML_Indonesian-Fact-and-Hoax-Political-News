# smsml_hoax_detection_project/Model/modelling.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# Load & augment dataset
def load_and_augment_dataset(path):
    df = pd.read_csv(path, compression='gzip')
    df['label'] = df['label'].astype(int)
    return df

# Tokenisasi
def tokenize_data(df):
    tokenizer = AutoTokenizer.from_pretrained("Muthiah192/distilbert-hoax-classifier")
    encodings = tokenizer(df['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    labels = df['label'].tolist()
    return encodings, labels

# Dataset
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

# Evaluasi
def evaluate_all(model, loader, name="Validation"):
    model.eval()
    true_labels, pred_labels = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return acc
