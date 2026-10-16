# Image for the Hugging Face Space (Docker SDK). tools/push_space.py uploads it as the Space's Dockerfile, next to
# app/, the code it imports, models/byt5-small-v4-real/ and data/rag/ (Wikipedia passages + e5 index).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/hf_cache HF_HUB_DISABLE_TELEMETRY=1 CPU_THREADS=2
WORKDIR /app
# HF Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt
# Bake the e5 query encoder into the image so a cold start downloads nothing
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')" \
    && chown -R user /app

COPY --chown=user . .
USER user

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
