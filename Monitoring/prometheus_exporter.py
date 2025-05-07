from prometheus_client import start_http_server, Gauge
import time
import psutil
from datetime import datetime

# Metric definitions
precision_non_hoax = Gauge('precision_non_hoax', 'Precision for non-hoax class')
recall_non_hoax = Gauge('recall_non_hoax', 'Recall for non-hoax class')
f1_non_hoax = Gauge('f1_non_hoax', 'F1-score for non-hoax class')

precision_hoax = Gauge('precision_hoax', 'Precision for hoax class')
recall_hoax = Gauge('recall_hoax', 'Recall for hoax class')
f1_hoax = Gauge('f1_hoax', 'F1-score for hoax class')

accuracy = Gauge('accuracy', 'Overall accuracy')
cpu_usage = Gauge('cpu_usage', 'CPU usage (%)')
memory_usage = Gauge('memory_usage', 'Memory usage (%)')

pred_non_hoax = Gauge('predicted_non_hoax', 'Total predicted non-hoax')
pred_hoax = Gauge('predicted_hoax', 'Total predicted hoax')

last_updated = Gauge('last_updated_timestamp', 'Timestamp of last metrics update')

def set_metrics():
    # Hardcoded values (you can fetch from actual logs or DB)
    precision_non_hoax.set(1.00)
    recall_non_hoax.set(1.00)
    f1_non_hoax.set(1.00)

    precision_hoax.set(0.99)
    recall_hoax.set(1.00)
    f1_hoax.set(0.99)

    accuracy.set(0.9974)

    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)

    pred_non_hoax.set(2058)
    pred_hoax.set(616)

    last_updated.set(time.time())

if __name__ == '__main__':
    start_http_server(8000)
    while True:
        set_metrics()
        time.sleep(60)
