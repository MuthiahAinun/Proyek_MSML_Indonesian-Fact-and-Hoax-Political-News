# smsml_hoax_detection_project/train_and_log.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, random_split
from modelling import IndoBertDataset, load_and_augment_dataset, tokenize_data, evaluate_all
import mlflow
import mlflow.transformers
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--data-path', type=str, default='./Experiment/preprocessing/dataset_cleaned_prepo.gz')
args = parser.parse_args()


# Load dataset
df = load_and_augment_dataset(args.data_path)
encodings, labels = tokenize_data(df)
dataset = IndoBertDataset(encodings, labels)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
_, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load pretrained model dan tokenizer dari Hugging Face
model_name = "Muthiah192/distilbert-hoax-classifier"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)

# MLflow Logging
mlflow.set_experiment("HoaxDetection")

with mlflow.start_run() as run:
    mlflow.log_param("source", "huggingface_pretrained")
    run_id = run.info.run_id

    # Simpan run_id ke file
    with open("last_run_id.txt", "w") as f:
        f.write(run_id)

    # Evaluasi dan log metric
    val_acc = evaluate_all(model, val_loader, name="Validation")
    test_acc = evaluate_all(model, test_loader, name="Test")
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    # Siapkan input_example (harus cocok dengan format serving)
    input_example = pd.DataFrame([{"text": "Berita ini mengandung unsur penipuan dan hoaks."}])

    # Log model ke MLflow
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model",
        input_example=input_example
    )

    print("Model dan metrik berhasil dilog ke MLflow.")
