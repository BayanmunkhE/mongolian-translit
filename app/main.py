"""FastAPI demo: informal Latin -> Cyrillic Mongolian (model vs. rule baseline) + Wikipedia search.

    uvicorn app.main:app --reload --port 7860
    MODEL_DIR=models/byt5-mn uvicorn app.main:app --port 7860          # local model
    MODEL_ID=<user>/byt5-small-mn-latin2cyrl uvicorn app.main:app      # from the HF Hub
    RAG_RETRIEVER=bm25 uvicorn app.main:app                            # e5 (default if indexed) | bm25 | off

Without a model the API still works and returns "model": null. POST /search needs data/rag/ (rag/corpus.py);
without it the endpoint returns 503 and the page hides the search card.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import get_baseline  # noqa: E402
from translit.candidates import candidates_mixed, word_alternatives  # noqa: E402
from translit.mixed import convert_mixed  # noqa: E402

log = logging.getLogger("uvicorn.error")
MAX_CHARS = 1000
STATIC = Path(__file__).parent / "static"

baseline = get_baseline(os.getenv("BASELINE", "mns5217"))
model = None
model_name = None
retriever = None
passages: list[dict] = []
# One generation at a time: on a small CPU host, parallel generate() calls fight over the same cores and every
# request gets slower. Queued requests wait here instead; repeated inputs hit the model's cache.
compute_lock = threading.Lock()
DEFAULT_MODELS = ("byt5-small-chat2", "byt5-small-chat", "byt5-small-v4-real", "byt5-mn")  # first present wins


@asynccontextmanager
async def lifespan(_app):
    load_model()
    load_retriever()
    yield


app = FastAPI(title="Mongolian Latin→Cyrillic transliteration", version="0.1.0", lifespan=lifespan)


def load_model():
    global model, model_name
    source = os.getenv("MODEL_DIR") or os.getenv("MODEL_ID")
    if not source:
        source = next((str(ROOT / "models" / m) for m in DEFAULT_MODELS if (ROOT / "models" / m).exists()), None)
    if not source:
        log.warning("No MODEL_DIR / MODEL_ID set: serving baseline only.")
        return
    try:
        if os.getenv("MODEL_BACKEND", "torch") == "onnx-tagger":
            # torch-free: the exported tagger (34 MB int8, ~2 ms per message) for small hosts
            from translit.onnx_backend import OnnxTaggerTransliterator
            model = OnnxTaggerTransliterator(source, threads=int(os.getenv("CPU_THREADS", "2")))
            model_name = os.getenv("MODEL_LABEL", f"ByT5 tagger ({Path(source).name}, ONNX int8)")
        else:
            from translit.model import Seq2SeqTransliterator
            model = Seq2SeqTransliterator(source, device=os.getenv("MODEL_DEVICE", "cpu"),
                                          num_beams=int(os.getenv("NUM_BEAMS", "4")),
                                          cpu_threads=int(os.getenv("CPU_THREADS", "0")) or None)
            model_name = os.getenv("MODEL_LABEL", f"ByT5-small ({Path(source).name})")
        model("sain baina uu")  # warm-up
        log.info("Loaded model from %s", source)
    except Exception as e:  # keep the demo up even if the model fails to load
        log.exception("Model load failed (%s); serving baseline only.", e)
        model = None


def load_retriever():
    global retriever, passages
    from rag.retriever import PASSAGES, RETRIEVERS, load_passages
    kind = os.getenv("RAG_RETRIEVER") or ("e5" if (PASSAGES.parent / "multilingual-e5-base.npy").exists() else "bm25")
    if kind == "off" or not PASSAGES.exists():
        log.warning("No RAG index (%s): /search disabled. Build it with rag/corpus.py.", PASSAGES)
        return
    try:
        passages = load_passages()
        retriever = RETRIEVERS[kind](passages)
        retriever.search(["монгол"], k=1)  # warm-up
        log.info("Loaded %s retriever over %d passages", kind, len(passages))
    except Exception as e:  # search is optional: keep transliteration up
        log.exception("Retriever load failed (%s); /search disabled.", e)
        retriever = None


class Request(BaseModel):
    text: str = Field(..., max_length=MAX_CHARS)


class Candidate(BaseModel):
    text: str
    prob: float


class Alternative(BaseModel):
    index: int  # word position in `model`
    options: list[str]


class Response(BaseModel):
    model: str | None
    baseline: str
    model_name: str | None
    baseline_name: str
    elapsed_ms: float
    candidates: list[Candidate] = []  # n-best, best first
    alternatives: list[Alternative] = []  # words of `model` where another candidate differs


@app.post("/transliterate", response_model=Response)
def transliterate(req: Request, n: int = 3):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    n = max(1, min(n, 5))
    t0 = time.perf_counter()
    cands: list[Candidate] = []
    with compute_lock:
        if model:
            cands = [Candidate(text=t, prob=round(p, 3)) for t, p in candidates_mixed(text, model, n)]
    out = cands[0].text if cands else None
    return Response(model=out, baseline=convert_mixed(text, baseline), model_name=model_name,
                    baseline_name="MNS 5217:2012 (дүрэм)",
                    elapsed_ms=round(1000 * (time.perf_counter() - t0), 1), candidates=cands,
                    alternatives=[Alternative(**a) for a in word_alternatives([c.text for c in cands])])


class Feedback(BaseModel):
    src: str = Field(..., max_length=MAX_CHARS)
    model: str = Field(..., max_length=4 * MAX_CHARS)  # what the model proposed
    chosen: str = Field(..., max_length=4 * MAX_CHARS)  # what the user picked


FEEDBACK = ROOT / "data" / "feedback" / "choices.jsonl"  # git-ignored; future training signal


@app.post("/feedback")
def feedback(fb: Feedback):
    """Store the user's word choice (only when it differs from the model's best)."""
    if fb.chosen.strip() == fb.model.strip():
        return {"stored": False}
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "src": fb.src, "model": fb.model,
                            "chosen": fb.chosen, "model_name": model_name}, ensure_ascii=False) + "\n")
    return {"stored": True}


class Hit(BaseModel):
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    converted: str
    converted_by: str
    latin_hits: list[Hit]
    converted_hits: list[Hit]
    elapsed_ms: float


@app.post("/search", response_model=SearchResponse)
def search(req: Request, k: int = 3):
    """Search Mongolian Wikipedia twice: with the raw Latin text and with its Cyrillic conversion."""
    if retriever is None:
        raise HTTPException(503, "search index not loaded")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    k = max(1, min(k, 10))
    t0 = time.perf_counter()
    with compute_lock:
        converted = convert_mixed(text, model if model else baseline)
        latin, cyr = retriever.search([text, converted], k=k)

    def hits(res):
        return [Hit(title=passages[i]["title"], text=passages[i]["text"][:300], score=round(s, 3)) for i, s in res]

    return SearchResponse(query=text, converted=converted, converted_by=model_name if model else "MNS 5217:2012",
                          latin_hits=hits(latin), converted_hits=hits(cyr),
                          elapsed_ms=round(1000 * (time.perf_counter() - t0), 1))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "search_loaded": retriever is not None}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
