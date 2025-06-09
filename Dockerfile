# Gunakan base image Python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependensi system
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements terlebih dahulu untuk caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi
COPY . .

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Jalankan app
CMD ["python", "app.py"]