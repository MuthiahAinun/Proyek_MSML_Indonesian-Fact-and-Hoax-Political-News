import pandas as pd
import re

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
    else:
        df.rename(columns={"Clean Narasi": "text"}, inplace=True)

# 3. Concat & cleanup
combined = pd.concat(dfs, ignore_index=True)
combined.dropna(subset=["text"], inplace=True)
combined.drop_duplicates(subset=["text", "hoax"], inplace=True)

# 4. Basic cleaning
def clean_text(t):
    t = re.sub(r"http\S+", "", str(t))
    t = re.sub(r"\s+", " ", t).strip()
    return t

combined["clean_text"] = combined["text"].apply(clean_text)

# 5. Filter length
combined["word_count"] = combined["clean_text"].apply(lambda x: len(x.split()))
combined = combined[(combined["word_count"] > 10) & (combined["word_count"] < 10000)]

# 6. Keep only needed cols
final = combined[["clean_text", "hoax"]].rename(columns={"clean_text":"text", "hoax":"label"})

# 7. Save processed dataset
final.to_csv("Experiment/preprocessing/dataset_cleaned.csv", index=False)
print(f"Processed {len(final)} rows.")
