# Example Prometheus exporter for additional metrics
from prometheus_client import start_http_server, Gauge
import time

# Metrik untuk Precision, Recall, F1-Score, Resource Utilization
precision_non_hoax = Gauge('precision_non_hoax', 'Precision for non-hoax class')
recall_non_hoax = Gauge('recall_non_hoax', 'Recall for non-hoax class')
f1_non_hoax = Gauge('f1_non_hoax', 'F1-Score for non-hoax class')
cpu_usage = Gauge('cpu_usage', 'CPU Usage in percentage')
memory_usage = Gauge('memory_usage', 'Memory Usage in percentage')

def set_metrics():
    # Update the metrics with actual values during the training or evaluation
    precision_non_hoax.set(0.99)  # example value
    recall_non_hoax.set(0.99)     # example value
    f1_non_hoax.set(0.99)         # example value
    cpu_usage.set(55)             # example value
    memory_usage.set(70)          # example value

if __name__ == '__main__':
    start_http_server(8000)  # Start Prometheus exporter at port 8000
    while True:
        set_metrics()
        time.sleep(60)  # Update metrics every 60 seconds
