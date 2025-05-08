FROM python:3.9-slim

WORKDIR /app

COPY . .

CMD ["echo", "This is a placeholder Docker image for hoax-exporter"]
