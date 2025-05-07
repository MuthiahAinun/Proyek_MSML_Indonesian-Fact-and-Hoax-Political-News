# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY Monitoring/prometheus_exporter.py .

CMD ["python", "prometheus_exporter.py"]
