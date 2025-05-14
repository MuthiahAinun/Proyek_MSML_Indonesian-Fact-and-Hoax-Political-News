import pandas as pd
import re
import string

# 1. Load raw datasets
paths = [
    "Experiment/dataset_raw/dataset_tempo_6k_cleaned.xlsx",
    "Experiment/dataset_raw/dataset_kompas_4k_cleaned.xlsx",
    "Experiment/dataset_raw/dataset_cnn_10k_cleaned.xlsx",
    "Experiment/dataset_raw/dataset_turnbackhoax_10_cleaned.xlsx",
]
dfs = [pd.read_excel(p) for p in paths]

# 2. Standardize text column
for df in dfs:
    if "text_new" in df.columns:
        df.rename(columns={"text_new": "text"}, inplace=True)
    elif "Clean Narasi" in df.columns:
        df.rename(columns={"Clean Narasi": "text"}, inplace=True)

# 3. Combine all datasets
combined = pd.concat(dfs, ignore_index=True)

# 4. Drop missing and duplicate values
combined.dropna(subset=["text"], inplace=True)
combined.drop_duplicates(subset=["text", "hoax"], inplace=True)

# 5. Clean text
def clean_text(text):
    text = str(text).lower()                                      # lowercase
    text = re.sub(r"http[s]?://\S+", "", text)                    # remove URLs
    text = re.sub(r"\d+", "", text)                               # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)                       # remove non-alphabetic chars
    text = re.sub(r"\s+", " ", text).strip()                      # normalize whitespace
    return text

combined["clean_text"] = combined["text"].apply(clean_text)

# 6. Word count filtering
combined["word_count"] = combined["clean_text"].apply(lambda x: len(x.split()))
filtered = combined[(combined["word_count"] >= 10) & (combined["word_count"] <= 1000)]

# 7. Select final columns
final = filtered[["clean_text", "hoax"]].rename(columns={"clean_text": "text", "hoax": "label"})

# 8. Save result
final.to_csv("Experiment/preprocessing/dataset_cleaned.csv", index=False)
print(f"✅ Dataset cleaned & saved: {len(final)} rows.")
