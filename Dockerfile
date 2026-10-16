# CPU image for Hugging Face Spaces (Docker SDK) or any container host.
#   docker build -t mn-translit .
#   docker run -p 7860:7860 -e MODEL_ID=<user>/byt5-small-mn-latin2cyrl mn-translit
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf
WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY baselines/ baselines/
COPY translit/ translit/
COPY app/ app/
# Optional: bake a local model into the image instead of downloading from the Hub
# COPY models/byt5-mn/ models/byt5-mn/

# HF Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
