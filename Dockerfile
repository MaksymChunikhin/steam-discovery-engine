# Inference-only image: serves the exported two-stage recommendation pipeline.
# No training dependencies (no torch / implicit / sentence-transformers).
# Образ только для инференса: отдаёт экспортированный двухстадийный пайплайн.
# Без обучающих зависимостей (нет torch / implicit / sentence-transformers).
FROM python:3.11-slim

WORKDIR /app

# System libs: libgomp1 (OpenMP) is required at runtime by LightGBM and faiss
# Системные либы: libgomp1 (OpenMP) нужен LightGBM и faiss в рантайме
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install inference dependencies only
# Ставим только серверные зависимости
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source, exported models and the data needed for serving
# Копируем код, экспортированные модели и данные, нужные для сервинга
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/games.parquet ./data/processed/games.parquet
COPY data/demo/ ./data/demo/

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
