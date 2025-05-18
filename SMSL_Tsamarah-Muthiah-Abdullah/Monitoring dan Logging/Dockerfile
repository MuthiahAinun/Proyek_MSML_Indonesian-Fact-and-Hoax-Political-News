FROM python:3.10

WORKDIR /app

COPY prometheus_exporter.py .
RUN pip install prometheus_client psutil

CMD ["python", "prometheus_exporter.py"]
