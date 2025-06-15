# modelling.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import mlflow
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
import joblib
from dotenv import load_dotenv

# -------------------------------
# DATA AUGMENTATION (EDA)
# -------------------------------
def random_deletion(words, p=0.1):
    if len(words) == 1:
        return words
    new_words = [word for word in words if random.uniform(0, 1) > p]
    if not new_words:
        new_words = [random.choice(words)]
    return new_words

def random_swap(words, n=1):
    new_words = words.copy()
    if len(new_words) < 2:
        return new_words
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def eda(text, num_aug=2):
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    if not words:
        return [text] * num_aug
    augmented = []
    for _ in range(num_aug):
        if random.choice(["swap", "delete"]) == "swap":
            new_words = random_swap(words, max(1, len(words)//10))
        else:
            new_words = random_deletion(words, p=0.1)
        augmented.append(" ".join(new_words))
    return augmented

def load_and_augment_dataset(path, augment=True, num_aug=2):
    df = pd.read_csv(path, compression='infer')
    df['label'] = df['label'].astype(int)
    if not augment:
        return df
    texts, labels = [], []
    for _, row in df.iterrows():
        texts.append(row['text'])
        labels.append(row['label'])
        for aug in eda(row['text'], num_aug=num_aug):
            texts.append(aug)
            labels.append(row['label'])
    return pd.DataFrame({'text': texts, 'label': labels})

# -------------------------------
# LOAD CONFIG
# -------------------------------
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("RF_CI_RemoteLog")

def train_and_log_model():
    df = load_and_augment_dataset("dataset_cleaned_prepo.gz", augment=True, num_aug=2)
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted', zero_division=0)
        recall = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("vectorizer", "tfidf-1000")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        joblib.dump(pipeline, "rf_model.pkl")
        mlflow.log_artifact("rf_model.pkl")

        input_example = pd.DataFrame(X_test[:1])
        input_example.to_csv("input_example.csv", index=False)
        mlflow.log_artifact("input_example.csv")

        report = classification_report(y_test, preds)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        print("\n✅ Training selesai dan model dicatat ke DagsHub.")

if __name__ == '__main__':
    train_and_log_model()
