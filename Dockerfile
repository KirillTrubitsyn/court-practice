# Fallback Docker образ. Railway по умолчанию подцепит nixpacks; используй Dockerfile,
# если хочешь воспроизводимости с локалкой.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Системные зависимости минимальны — pymorphy3 чисто питонячий.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
# Дефолтные индекс и эмбеддинги. При первом старте сервер скопирует их
# в Railway Volume через _seed_volume_if_empty().
COPY data ./data

# Volume mountpoint должен существовать в образе (Railway монтирует поверх).
RUN mkdir -p /data

ENV DATA_DIR=/data
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
