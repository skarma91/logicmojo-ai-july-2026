# The tax research team as a container: same image locally and on Render's free tier.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.fastembed

WORKDIR /app

# Dependencies first, in their own layer: code edits do not reinstall packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the embedding model and the corpus vectors into the image, so the running
# service never downloads anything and only embeds queries.
RUN python build_index.py

# A non-root user: the service never needs to write outside /app and /tmp.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# Render sets PORT; locally it defaults to 8000. One worker: 512 MB holds one copy of the model.
CMD ["sh", "-c", "uvicorn service:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
