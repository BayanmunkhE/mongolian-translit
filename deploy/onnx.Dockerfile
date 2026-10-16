# Tiny image for free hosts (Render / Cloud Run / Fly): the FastAPI demo with the ONNX tagger, no torch.
# ~150 MB image, ~200 MB RAM. Wikipedia search is off (its e5 model needs torch); the UI hides the search card.
#   docker build -f deploy/onnx.Dockerfile -t mn-translit-onnx .
#   docker run -p 7860:7860 mn-translit-onnx
FROM python:3.12-slim
WORKDIR /srv
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic onnxruntime numpy
COPY app app
COPY baselines baselines
COPY translit translit
COPY models/onnx/tagger-4e-full/tagger.int8.onnx models/onnx/tagger/tagger.int8.onnx
COPY models/onnx/tagger-4e-full/vocab.json models/onnx/tagger/vocab.json
ENV MODEL_BACKEND=onnx-tagger MODEL_DIR=models/onnx/tagger RAG_RETRIEVER=off PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
