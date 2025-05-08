from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time

app = FastAPI()

# Load model from Hugging Face
MODEL_NAME = "Muthiah192/distilbert-hoax-classifier"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Prometheus metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total number of prediction requests")
PREDICT_HOAX = Counter("predicted_hoax_total", "Total hoax predictions")
PREDICT_NON_HOAX = Counter("predicted_non_hoax_total", "Total non-hoax predictions")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Time taken for a prediction")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
@INFERENCE_LATENCY.time()
def predict(request: TextRequest):
    REQUEST_COUNT.inc()
    inputs = tokenizer(request.text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label = "hoax" if pred == 1 else "non-hoax"
    if pred == 1:
        PREDICT_HOAX.inc()
    else:
        PREDICT_NON_HOAX.inc()

    return {
        "text": request.text,
        "label": label,
        "confidence": round(probs[0][pred].item(), 4)
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    return {"message": "Hoax detection inference service"}
