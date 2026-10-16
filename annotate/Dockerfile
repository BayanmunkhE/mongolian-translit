# Annotation UI for the real test set (no torch). Build from the repo root:
#   docker compose up annotate          -> http://localhost:8000
# data/ is mounted as a volume, so data/annotate.db stays on the host.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements-annotate.txt .
RUN pip install -r requirements-annotate.txt

COPY baselines/ baselines/
COPY tools/ tools/
COPY generate_pairs.py .
COPY annotate/ annotate/

EXPOSE 8000
CMD ["uvicorn", "annotate.main:app", "--host", "0.0.0.0", "--port", "8000"]
