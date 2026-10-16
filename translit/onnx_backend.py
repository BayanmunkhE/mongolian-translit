"""Torch-free inference for the web app: the exported tagger (tools/export_tagger_onnx.py) behind the same
interface as Seq2SeqTransliterator (.transliterate / .batch / .nbest / .name), so the demo can run on a
256 MB host with onnxruntime + numpy only (MODEL_BACKEND=onnx-tagger MODEL_DIR=models/onnx/tagger-4e).

The tagger gives one answer per input, so nbest() returns a single candidate; the pick-a-word UI then shows no
alternatives. Inputs longer than the training length are split on whitespace into <= max_bytes chunks.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

UNK = "<unk>"


def encode(text: str) -> list[int]:
    return [b + 3 for b in text.encode("utf-8")] + [1]


def decode(src: str, pred_ids, vocab: list[str]) -> str:
    out, pos = [], 0
    for ch in src:
        n = len(ch.encode("utf-8"))
        lab = vocab[pred_ids[pos]] if pos < len(pred_ids) else UNK
        out.append(ch if lab == UNK else lab)
        pos += n
    return "".join(out)


class OnnxTaggerTransliterator:
    def __init__(self, model_dir: str, threads: int = 2, int8: bool = True, max_bytes: int = 200):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        d = Path(model_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        name = "tagger.int8.onnx" if int8 and (d / "tagger.int8.onnx").exists() else "tagger.onnx"
        self.sess = ort.InferenceSession(str(d / name), so, providers=["CPUExecutionProvider"])
        self.vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        self.name = d.name
        self.max_bytes = max_bytes
        self._cached = lru_cache(maxsize=4096)(self._convert)

    def _chunks(self, text: str) -> list[str]:
        if len(text.encode()) <= self.max_bytes:
            return [text]
        out, cur = [], ""
        for w in text.split():
            cand = f"{cur} {w}".strip()
            if cur and len(cand.encode()) > self.max_bytes:
                out.append(cur)
                cur = w
            else:
                cur = cand
        return out + ([cur] if cur else [])

    def _convert(self, text: str) -> str:
        parts = []
        for chunk in self._chunks(text):
            ids = self.np.array([encode(chunk)], dtype=self.np.int64)
            pred = self.sess.run(None, {"input_ids": ids, "attention_mask": self.np.ones_like(ids)})[0][0]
            parts.append(decode(chunk, pred.tolist(), self.vocab))
        return " ".join(parts)

    def transliterate(self, text: str) -> str:
        text = text.strip()
        return "\n".join(self._cached(line) if line.strip() else "" for line in text.split("\n")) if text else ""

    __call__ = transliterate

    def batch(self, texts: list[str], batch_size: int = 32) -> list[str]:
        return [self.transliterate(t) for t in texts]

    def nbest(self, text: str, n: int = 3) -> list[tuple[str, float]]:
        return [(self.transliterate(text), 1.0)]
