# The Anchor Paper RAG API.
# Build: docker build -t anchor-rag:v2.0 .
# Run:   docker run --env-file .env -p 8000:8000 anchor-rag:v2.0
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANCHOR_RAG_CONFIG=serving \
    PORT=8000

WORKDIR /app

# Installed before the code is copied, so a code change does not reinstall every package.
# Only the packages the API needs are installed: evaluation and PDF tools stay out of the image.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ src/
COPY configs/ configs/
COPY prompts/ prompts/
COPY data/index_manifests/ data/index_manifests/

# The passages are not in the image. On start, the API rebuilds them from Pinecone and checks them
# against data/index_manifests, so it runs as a user that can write the artifacts folder.
RUN useradd --create-home app && mkdir -p artifacts && chown -R app /app
USER app

EXPOSE 8000
# Rebuilding the passages takes a little while on the first start, hence the start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"

# Hosting platforms set PORT. exec lets uvicorn receive the stop signal and shut down cleanly.
CMD ["sh", "-c", "exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
