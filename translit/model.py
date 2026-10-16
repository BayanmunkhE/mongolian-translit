"""Inference wrapper for the fine-tuned seq2seq transliterator (ByT5 / mT5).

Long inputs are split into segments (lines -> sentences -> word groups) that
fit the training length, transliterated in batches, and re-joined.
"""
from __future__ import annotations

import re
from functools import lru_cache

SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_segments(text: str, max_bytes: int) -> list[str]:
    """Split one line into pieces of <= max_bytes UTF-8 bytes on natural boundaries."""
    pieces = []
    for sent in SENT_RE.split(text.strip()):
        if len(sent.encode()) <= max_bytes:
            pieces.append(sent)
            continue
        cur = ""
        for w in sent.split():
            cand = f"{cur} {w}".strip()
            if cur and len(cand.encode()) > max_bytes:
                pieces.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            pieces.append(cur)
    return [p for p in pieces if p]


class Seq2SeqTransliterator:
    def __init__(self, model_dir: str, device: str | None = None, num_beams: int = 4,
                 max_src_bytes: int = 200, max_new_tokens: int = 512, cpu_threads: int | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        if cpu_threads:
            torch.set_num_threads(cpu_threads)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device).eval()
        self.name = model_dir.rstrip("/\\").split("/")[-1].split("\\")[-1]
        self.num_beams = num_beams
        self.max_src_bytes = max_src_bytes
        self.max_new_tokens = max_new_tokens
        self._cached = lru_cache(maxsize=2048)(self._transliterate_uncached)

    # -- public API ---------------------------------------------------------
    def transliterate(self, text: str) -> str:
        return self._cached(text)

    __call__ = transliterate

    def batch(self, texts: list[str], batch_size: int = 32) -> list[str]:
        """Transliterate many texts; segments from all texts are batched together."""
        seg_index, segments = [], []  # (text_idx, line_idx) for each segment
        layouts = []
        for ti, text in enumerate(texts):
            lines = text.split("\n")
            layouts.append([0] * len(lines))
            for li, line in enumerate(lines):
                for seg in split_segments(line, self.max_src_bytes):
                    seg_index.append((ti, li))
                    segments.append(seg)
                    layouts[ti][li] += 1
        outputs = self._generate(segments, batch_size)

        rebuilt: list[list[list[str]]] = [[[] for _ in lay] for lay in layouts]
        for (ti, li), out in zip(seg_index, outputs):
            rebuilt[ti][li].append(out)
        return ["\n".join(" ".join(parts) for parts in lines) for lines in rebuilt]

    def nbest(self, text: str, n: int = 3) -> list[tuple[str, float]]:
        """Top-n beam candidates for a short text with their relative probabilities (softmax of the beam scores).
        Texts that need segmenting fall back to the single best conversion."""
        text = text.strip()
        if not text or "\n" in text or len(split_segments(text, self.max_src_bytes)) != 1:
            return [(self.transliterate(text), 1.0)]
        enc = self.tokenizer([text], return_tensors="pt", truncation=True,
                             max_length=self.max_src_bytes + 8).to(self.device)
        max_new = min(self.max_new_tokens, int(len(text.encode()) * 2.2) + 16)
        beams = max(self.num_beams, n)
        with self.torch.inference_mode():
            out = self.model.generate(**enc, num_beams=beams, num_return_sequences=n, max_new_tokens=max_new,
                                      early_stopping=True, output_scores=True, return_dict_in_generate=True)
        texts = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        # sequences_scores are length-normalised log-probs; undo that so the softmax reflects real odds
        lengths = (out.sequences != self.tokenizer.pad_token_id).sum(1).float()
        probs = self.torch.softmax(out.sequences_scores.float() * lengths, dim=0).tolist()
        merged: dict[str, list] = {}  # key: case-folded text; beams that differ only in casing are one candidate
        for t, p in zip(texts, probs):
            key = t.strip().lower()
            if key in merged:
                merged[key][1] += p
            else:
                merged[key] = [t.strip(), p]
        return sorted((tuple(v) for v in merged.values()), key=lambda kv: -kv[1])

    # -- internals ----------------------------------------------------------
    def _transliterate_uncached(self, text: str) -> str:
        return self.batch([text])[0]

    def _generate(self, segments: list[str], batch_size: int) -> list[str]:
        results = [""] * len(segments)
        order = sorted(range(len(segments)), key=lambda i: len(segments[i]))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = [segments[i] for i in idx]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                 max_length=self.max_src_bytes + 8).to(self.device)
            # Cyrillic is 2 bytes/char in UTF-8: allow ~2.2x the source length.
            longest = max(len(s.encode()) for s in batch)
            max_new = min(self.max_new_tokens, int(longest * 2.2) + 16)
            with self.torch.inference_mode():
                out = self.model.generate(**enc, num_beams=self.num_beams,
                                          max_new_tokens=max_new, early_stopping=True)
            decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            for i, d in zip(idx, decoded):
                results[i] = d.strip()
        return results
