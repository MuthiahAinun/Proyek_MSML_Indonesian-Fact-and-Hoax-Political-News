from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.sklearn
import time
from prometheus_client import Counter, Histogram, start_http_server
import os

app = FastAPI()

# Load model from inside the Docker image
model = mlflow.sklearn.load_model("rf_model")  # sesuai path di dalam image

# Prometheus metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total number of inference requests')
PREDICTION_LABEL = Counter('predicted_label_count', 'Count of predicted labels', ['label'])
REQUEST_LATENCY = Histogram('inference_request_latency_seconds', 'Latency per request')

class InputText(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    start_http_server(8000)  # expose Prometheus metrics on port 8000 (in-container)

@app.post("/predict")
@REQUEST_LATENCY.time()
def predict(input_data: InputText):
    REQUEST_COUNT.inc()

    prediction = model.predict([input_data.text])[0]
    PREDICTION_LABEL.labels(label=str(prediction)).inc()

    return {"text": input_data.text, "prediction": int(prediction)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)