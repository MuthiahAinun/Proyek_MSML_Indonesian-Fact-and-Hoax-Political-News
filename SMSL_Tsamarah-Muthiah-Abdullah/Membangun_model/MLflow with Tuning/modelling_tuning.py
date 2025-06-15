import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow
from dotenv import load_dotenv
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
mlflow.set_tracking_uri("file:./mlruns2")
mlflow.set_experiment("RF_Tuning_ManualLog")

df = load_and_augment_dataset("dataset_cleaned_prepo.gz", augment=True, num_aug=2)
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [None, 10, 20]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)

with mlflow.start_run():
    grid.fit(X_train, y_train)
    preds = grid.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(grid.best_estimator_, "model")
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)


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
