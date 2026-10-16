#!/usr/bin/env python3
"""Build the RAG knowledge base: Mongolian Wikipedia articles split into ~100-word passages.

    python rag/corpus.py                        # -> data/rag/passages.jsonl
    python rag/corpus.py --max-articles 2000    # quick test

Each line: {"id": "<article id>-<n>", "title": ..., "text": ...}.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CYRILLIC = re.compile(r"[А-Яа-яЁёӨөҮү]")
SENT_END = re.compile(r"(?<=[.!?])\s+")


def chunk(text: str, target: int = 100, max_words: int = 160) -> list[str]:
    """Greedy packing of whole paragraphs up to ~target words; longer paragraphs are split at sentence ends."""
    units = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) <= max_words:
            units.append(para)
        else:
            units.extend(s for s in SENT_END.split(para) if s.strip())
    passages, cur, n = [], [], 0
    for unit in units:
        w = len(unit.split())
        if cur and n + w > target:
            passages.append(" ".join(cur))
            cur, n = [], 0
        cur.append(unit)
        n += w
    if cur:
        passages.append(" ".join(cur))
    return passages


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="wikimedia/wikipedia")
    ap.add_argument("--config", default="20231101.mn")
    ap.add_argument("--max-articles", type=int, default=0)
    ap.add_argument("--min-words", type=int, default=8, help="drop shorter passages (stubs, list debris)")
    ap.add_argument("--out", default="data/rag/passages.jsonl")
    args = ap.parse_args(argv)

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.config, split="train")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    n_art = n_pass = 0
    with out.open("w", encoding="utf-8") as f:
        for art in ds:
            if args.max_articles and n_art >= args.max_articles:
                break
            n_art += 1
            for i, text in enumerate(chunk(art["text"])):
                if len(text.split()) < args.min_words or not CYRILLIC.search(text):
                    continue
                f.write(json.dumps({"id": f"{art['id']}-{i}", "title": art["title"], "text": text},
                                   ensure_ascii=False) + "\n")
                n_pass += 1
    print(f"{n_art} articles -> {n_pass} passages in {out}")


if __name__ == "__main__":
    main()
