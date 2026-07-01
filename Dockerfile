# Microgrid Multi-Agent System — pipeline image
FROM python:3.13-slim

# System deps occasionally needed by scientific wheels (kept minimal).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code, data and pre-trained models.
COPY src ./src
COPY data ./data
COPY models ./models

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MICROGRID_DB_URL=sqlite:////app/results/microgrid.db

# Default: run the orchestrated pipeline once.
CMD ["python", "-m", "src.pipeline.flow", "--timestamp", "2017-06-15 19:00:00"]
