#!/usr/bin/env python3
"""CPU latency of a fine-tuned model per message, as the demo/API would run it.

    python eval/benchmark_cpu.py --model models/byt5-mn-v3 --beams 1 4 8
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from translit.model import Seq2SeqTransliterator  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--beams", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--messages", default="data/real_test.jsonl", help="JSONL with a 'src' field (inputs only)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args(argv)

    with open(args.messages, encoding="utf-8") as f:
        texts = [json.loads(l)["src"] for l in f if l.strip()][:args.n]
    print(f"{len(texts)} messages, avg {sum(map(len, texts)) / len(texts):.0f} chars, model {args.model}")
    print("| Beams | median ms/message | p90 ms/message |\n|---:|---:|---:|")
    for beams in args.beams:
        tr = Seq2SeqTransliterator(args.model, device="cpu", num_beams=beams, cpu_threads=args.threads or None)
        tr.batch(texts[:2])  # warm-up
        times = []
        for t in texts:
            t0 = time.perf_counter()
            tr.batch([t])  # batch() bypasses the cache, like a new message
            times.append(1000 * (time.perf_counter() - t0))
        times.sort()
        print(f"| {beams} | {statistics.median(times):.0f} | {times[int(0.9 * (len(times) - 1))]:.0f} |")


if __name__ == "__main__":
    main()
