FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy full project
COPY . .

# Set working directory to backend
WORKDIR /app/backend

EXPOSE 5000

CMD ["python", "app.py"]