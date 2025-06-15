import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os


import random
import re

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
    import pandas as pd
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


load_dotenv()
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("RF_Local_AutoLog")
mlflow.sklearn.autolog()

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

    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    report = classification_report(y_test, preds)
    print(report)
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
