# The arXiv research team as a container: the same image runs locally and on Render's free tier.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.fastembed

WORKDIR /app

# Dependencies first, in their own layer: code edits do not reinstall packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the embedding model and the corpus vectors into the image: the running service
# never downloads anything and only embeds queries. Uses data/corpus.jsonl if you
# fetched it before building, otherwise the committed example corpus.
RUN python -c "import retriever; print(retriever.load_index(), 'papers indexed')"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# Render sets PORT. One worker: 512 MB holds one copy of the embedding model.
CMD ["sh", "-c", "uvicorn service:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
