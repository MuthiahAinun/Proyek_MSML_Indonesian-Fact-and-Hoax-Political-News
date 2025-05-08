# Proyek_MSML_Indonesian-Fact-and-Hoax-Political-News

## 🗂️ Dataset Description: **Indonesian Fact and Hoax Political News**

This dataset is a collection of Indonesian political news articles categorized into two main classes: Non-Hoax/Fact (valid) and Hoax (disinformation). It is sourced from both credible and non-credible platforms and is intended for training text classification models to detect fake news.

**📁 Dataset Files**

The dataset consists of the following files and their respective sources:

- `dataset_tempo_6k_cleaned.xlsx` – Valid political news from Tempo (~6,000 entries)

- `dataset_kompas_4k_cleaned.xlsx` – Valid political news from Kompas (~4,000 entries)

- `dataset_cnn_10k_cleaned.xlsx` – Valid political news from CNN Indonesia (~10,000 entries)

- `dataset_turnbackhoax_10_cleaned.xlsx` – Hoax news from Turnbackhoax.id (~10,000 entries)

**✅ Truth Labels**

1. **Valid / Factual:** Collected from mainstream and trusted news portals: CNN Indonesia, Tempo, and Kompas.

2. **Hoax / Disinformation:** Collected from Turnbackhoax.id, a site that aggregates and verifies false or misleading claims.

**🔍 Dataset Source**

This dataset was downloaded from Kaggle:  
[Indonesian Fact and Hoax Political News](https://www.kaggle.com/datasets/linkgish/indonesian-fact-and-hoax-political-news?resource=download)

---
## ⚙️ Project Workflow Overview (GitHub Actions)

This project is designed to be built and executed using **GitHub Actions workflows**. 

### Step 1️⃣ : Running the Preprocessing Pipeline Workflow (`preprocess.yml`)
The first step in the process is **preprocessing the four datasets**. Simply trigger the workflow defined in the `preprocess.yml` file located in `.github/workflows/`.

Once the workflow runs successfully, the **preprocessed dataset will be saved as an artifact**, which can be downloaded directly from the GitHub Actions interface.  
📎 Artifact Example:  
![Preprocessing Artifact](Experiment/preprocessing/Artifak-Preprocessing.png)

---

### 📁 Project Folder Structure

```
Proyek_MSML_Indonesian-Fact-and-Hoax-Political-News
├── .github/workflows
│ └── preprocess.yml # Workflow file for dataset preprocessing
├── Experiment
│ ├── dataset_raw # Folder containing original raw datasets
│ │ ├── dataset_cnn_10k_cleaned.xlsx
│ │ ├── dataset_kompas_4k_cleaned.xlsx
│ │ ├── dataset_tempo_6k_cleaned.xlsx
│ │ └── dataset_turnbackhoax_10k_cleaned.xlsx
│ ├── preprocessing # Folder for all preprocessing outputs
│ │ ├── Artifak-Preprocessing.png # Image preview of preprocessing artifact
│ │ ├── Eksperimen_MSML_Tsamarah_Muthiah_Abdullah.ipynb # Full notebook for preprocessing, training, and inference (Colab-based)
│ │ ├── automate_Tsamarah-Muthiah-Abdullah.py # Python script to automate preprocessing (used in preprocess.yml)
│ │ ├── dataset-cleaned.gz # Cleaned dataset exported from Colab
│ │ └── dataset_cleaned.gz # Cleaned dataset generated automatically from GitHub workflow artifact
```

---

> This setup ensures reproducibility and automation of the preprocessing phase, making it easier to integrate into continuous workflows and model training pipelines.

---
### Step 2️⃣ : Running the CI Workflow (`ci.yaml`)

The second step of this project involves running the **CI workflow file `ci.yaml`**, which automates the training and evaluation process. This workflow will trigger the execution of `train_and_log.py` and `modelling.py`. The key processes in this step include:

- Performing **oversampling using Easy Data Augmentation (EDA)** to address class imbalance (minority class: `hoax == 1`)
- Tokenizing the text data
- Setting up PyTorch DataLoaders
- Splitting the dataset into 80% training and 20% validation
- Loading a **pre-trained model** from Hugging Face:  
  🔗 [Hugging Face Model - distilbert-hoax-classifier](https://huggingface.co/Muthiah192/distilbert-hoax-classifier/tree/main)
- Evaluating the model on the dataset
- Uploading MLflow artifacts
- Building and pushing a Docker image to Docker Hub:  
  🔗 [Docker Image - Hoax Exporter](https://hub.docker.com/r/muthiah192/hoax-exporter)

📎 Artifact Example:  
![Model Artifact](Experiment/preprocessing/Artifak-Model.png)

---

### 📁 Folder Structure for Step 2
```
├── .github/workflows
│ └── ci.yaml # CI workflow file for model training and Docker image build
├── Model
│ ├── Artifak-Model.png # Visual artifact from CI workflow run
│ ├── Dashboard-Monitoring-Grafana-12-metrics.png # Grafana monitoring dashboard preview
│ ├── MLProject # MLflow project file to enable automated retraining
│ ├── URL_Docker_Image # File containing link to the generated Docker image
│ ├── URL_Model_Saved # File containing link to the saved Hugging Face model
│ ├── augment.py # Script to perform text augmentation (EDA)
│ ├── conda.yaml # MLflow environment specification file (see below for description)
│ ├── modelling.py # Contains model architecture, dataset splitting, and tokenization logic
│ └── train_and_log.py # Script to train the model, evaluate it, and log metrics to MLflow
├── Dockerfile # Dockerfile to build the inference/exporter image
```

---

### 📄 File Descriptions

- **`conda.yaml`**: Defines the environment for MLflow to ensure reproducibility. Includes dependencies such as `pytorch`, `transformers`, `scikit-learn`, and `mlflow`.
- **`modelling.py`**: Contains the full training pipeline including:
  - Easy Data Augmentation (EDA) for oversampling
  - Tokenization using Hugging Face tokenizer
  - Dataset creation and splitting
  - Model architecture definition
- **`train_and_log.py`**: Runs the training process, evaluates the model, and logs metrics and artifacts to MLflow. This script is invoked directly by the `ci.yaml` workflow.

---

### 📊 Model Evaluation Results

#### ✅ **Validation Set Results**
- **Accuracy**: 0.9970

          precision    recall  f1-score   support

non-hoax       1.00      1.00      1.00      2090
    hoax       0.99      1.00      0.99       584

accuracy                           1.00      2674

macro avg 0.99 1.00 1.00 2674
weighted avg 1.00 1.00 1.00 2674

Prediction distribution: Counter({0: 2086, 1: 584})
Actual label distribution: Counter({0: 2090, 1: 584})


#### ✅ **Test Set Results**
- **Accuracy**: 0.9966

          precision    recall  f1-score   support

non-hoax       1.00      1.00      1.00      2056
    hoax       0.99      1.00      0.99       618

accuracy                           1.00      2674

macro avg 0.99 1.00 1.00 2674
weighted avg 1.00 1.00 1.00 2674

Prediction distribution: Counter({0: 2051, 1: 623})
Actual label distribution: Counter({0: 2056, 1: 618})


---

> This step is critical for ensuring model performance and deploying the result into a containerized environment for inference or monitoring purposes.
---



