from prometheus_client import start_http_server, Gauge
import time
import psutil
import json
import os
from datetime import datetime

# Prometheus Gauges
precision_non_hoax = Gauge('precision_non_hoax', 'Precision for non-hoax class (label 0)')
recall_non_hoax = Gauge('recall_non_hoax', 'Recall for non-hoax class (label 0)')
f1_non_hoax = Gauge('f1_non_hoax', 'F1-score for non-hoax class (label 0)')

precision_hoax = Gauge('precision_hoax', 'Precision for hoax class (label 1)')
recall_hoax = Gauge('recall_hoax', 'Recall for hoax class (label 1)')
f1_hoax = Gauge('f1_hoax', 'F1-score for hoax class (label 1)')

accuracy = Gauge('accuracy', 'Overall accuracy of the model')
cpu_usage = Gauge('cpu_usage_percent', 'Current CPU usage (%)')
memory_usage = Gauge('memory_usage_percent', 'Current memory usage (%)')
last_updated = Gauge('last_metrics_update_time', 'Timestamp of last metrics update')

# Tambahan: Total updates & durasi update
update_counter = Gauge('update_count', 'Total number of times the metrics have been updated')
update_duration = Gauge('update_duration_seconds', 'Duration it took to load and update metrics')

def load_metrics(path='classification_metrics.json'):
    if not os.path.exists(path):
        print(f"❌ File '{path}' tidak ditemukan.")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def update_metrics():
    start_time = time.time()

    metrics = load_metrics()
    if metrics:
        # Class 0: non-hoax
        precision_non_hoax.set(metrics.get("0", {}).get("precision", 0))
        recall_non_hoax.set(metrics.get("0", {}).get("recall", 0))
        f1_non_hoax.set(metrics.get("0", {}).get("f1-score", 0))

        # Class 1: hoax
        precision_hoax.set(metrics.get("1", {}).get("precision", 0))
        recall_hoax.set(metrics.get("1", {}).get("recall", 0))
        f1_hoax.set(metrics.get("1", {}).get("f1-score", 0))

        # Accuracy
        accuracy.set(metrics.get("accuracy", 0))

        # System metrics
        cpu_usage.set(psutil.cpu_percent(interval=1))
        memory_usage.set(psutil.virtual_memory().percent)

        # Update time and count
        last_updated.set(time.time())
        update_counter.inc()

        duration = time.time() - start_time
        update_duration.set(duration)

        print(f"✅ Metrics updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({duration:.2f}s)")

if __name__ == "__main__":
    print("🚀 Starting Prometheus exporter on port 8000...")
    start_http_server(8000)
    while True:
        update_metrics()
        time.sleep(60)  # update every 60 seconds
