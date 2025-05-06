import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model dan tokenizer dari folder yang sudah disimpan
MODEL_PATH = "saved_model_hoax"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Muat tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Fungsi inferensi
def predict(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return {
        "text": text,
        "label": "hoax" if pred == 1 else "non-hoax",
        "confidence": round(probs[0][pred].item(), 4)
    }

# Contoh penggunaan
if __name__ == "__main__":
    test_texts = [
        "Presiden Jokowi meluncurkan program vaksin nasional gratis.",
        "Presiden Jokowi akan membagikan dana bantuan melalui WhatsApp."
    ]
    for text in test_texts:
        result = predict(text)
        print(f"\nTeks: {result['text']}")
        print(f"Prediksi: {result['label']}")
        print(f"Confidence: {result['confidence']}")
