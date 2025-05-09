# smsml_hoax_detection_project/train_and_log.py

import torch
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, random_split
from modelling import IndoBertDataset, load_and_augment_dataset, tokenize_data, evaluate_all
import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = load_and_augment_dataset("Experiment/preprocessing/dataset_cleaned.gz")
encodings, labels = tokenize_data(df)
dataset = IndoBertDataset(encodings, labels)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
_, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load pretrained model dari Hugging Face
model = AutoModelForSequenceClassification.from_pretrained("Muthiah192/distilbert-hoax-classifier")
model.to(device)

# MLflow Logging
mlflow.set_experiment("HoaxDetection")

with mlflow.start_run():
    mlflow.log_param("source", "huggingface_pretrained")
    val_acc = evaluate_all(model, val_loader, name="Validation")
    test_acc = evaluate_all(model, test_loader, name="Test")
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("test_accuracy", test_acc)
