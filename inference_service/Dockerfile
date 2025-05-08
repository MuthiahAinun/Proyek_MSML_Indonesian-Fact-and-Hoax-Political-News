FROM python:3.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy kode FastAPI
COPY . .

# Jalankan server
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8001"]
