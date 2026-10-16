"""Search the Wikipedia passages (rag/corpus.py) with BM25 or dense multilingual embeddings.

Indexes are built on first use and cached next to the passages (data/rag/). Both retrievers return,
per query, a list of (passage index, score) with score > 0, best first: a query whose words never occur
in the corpus gets an empty list instead of arbitrary passages.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PASSAGES = ROOT / "data" / "rag" / "passages.jsonl"
WORD = re.compile(r"\w+")


def load_passages(path: Path = PASSAGES) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def doc_text(p: dict) -> str:
    return f"{p['title']}. {p['text']}"


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self, passages: list[dict], cache_dir: Path = PASSAGES.parent):
        import bm25s
        path = cache_dir / "bm25"
        if path.exists():
            self.index = bm25s.BM25.load(str(path))
        else:
            self.index = bm25s.BM25()
            self.index.index([tokenize(doc_text(p)) for p in passages], show_progress=False)
            self.index.save(str(path))
        self.vocab = self.index.vocab_dict

    def search(self, queries: list[str], k: int = 10) -> list[list[tuple[int, float]]]:
        results = []
        for q in queries:
            tokens = [t for t in tokenize(q) if t in self.vocab]
            if not tokens:
                results.append([])
                continue
            docs, scores = self.index.retrieve([tokens], k=k, show_progress=False)
            results.append([(int(d), float(s)) for d, s in zip(docs[0], scores[0]) if s > 0])
        return results


class DenseRetriever:
    """multilingual-e5 embeddings, cosine similarity (exact search; 90k passages fit in memory)."""
    name = "e5"

    def __init__(self, passages: list[dict], cache_dir: Path = PASSAGES.parent,
                 model: str = "intfloat/multilingual-e5-base", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device)
        path = cache_dir / f"{model.split('/')[-1]}.npy"
        if path.exists():
            self.emb = np.load(path).astype(np.float32)
        else:
            emb = self.model.encode([f"passage: {doc_text(p)}" for p in passages], batch_size=128,
                                    normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
            np.save(path, emb.astype(np.float16))
            self.emb = emb.astype(np.float32)

    def search(self, queries: list[str], k: int = 10) -> list[list[tuple[int, float]]]:
        q = self.model.encode([f"query: {x}" for x in queries], batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
        scores = q @ self.emb.T
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        results = []
        for row, idx in zip(scores, top):
            idx = idx[np.argsort(-row[idx])]
            results.append([(int(i), float(row[i])) for i in idx if row[i] > 0])
        return results


RETRIEVERS = {"bm25": BM25Retriever, "e5": DenseRetriever}
