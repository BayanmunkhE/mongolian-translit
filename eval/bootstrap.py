#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for CER and exact match from an evaluate.py predictions CSV.

    python eval/bootstrap.py results/v2/pred_real_test.csv
    python eval/bootstrap.py results/v2/pred_real_test.csv --systems byt5-v1 byt5-v2 --n 5000

Resamples sentences with replacement (the same resample for every system, so differences are
paired) and reports 95% intervals. Uses the corpus CER of eval/metrics.py: total edits / total
reference characters after norm().
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import edit_distance, norm  # noqa: E402


def load(path, refs_path=None) -> tuple[list[str], dict[str, list[tuple[int, int, int]]]]:
    """Per system, one (edits, ref_chars, exact) tuple per sentence.

    refs_path: JSONL whose tgt (matched by src) replaces the CSV's tgt, e.g. after references were corrected.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if refs_path:
        with open(refs_path, encoding="utf-8") as f:
            new = {d["src"]: d["tgt"] for d in map(json.loads, f) if d}
        for r in rows:
            r["tgt"] = new[r["src"]]
    systems = [c[:-5] for c in rows[0] if c.endswith("_pred")]
    stats = {s: [] for s in systems}
    for r in rows:
        ref = norm(r["tgt"])
        for s in systems:
            hyp = norm(r[f"{s}_pred"])
            stats[s].append((edit_distance(ref, hyp), len(ref), int(ref == hyp)))
    return systems, stats


def corpus(stats, idx) -> tuple[float, float]:
    edits = sum(stats[i][0] for i in idx)
    chars = sum(stats[i][1] for i in idx)
    return edits / max(1, chars), sum(stats[i][2] for i in idx) / max(1, len(idx))


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def bootstrap(stats: dict, systems: list[str], n: int, seed: int):
    """Returns per-system (cer, em, cer_lo, cer_hi, em_lo, em_hi) and resampled CERs for pairing."""
    size = len(next(iter(stats.values())))
    rng = random.Random(seed)
    samples = [[rng.randrange(size) for _ in range(size)] for _ in range(n)]
    full = list(range(size))
    summary, cers = {}, {}
    for s in systems:
        cer, em = corpus(stats[s], full)
        res = [corpus(stats[s], idx) for idx in samples]
        cers[s] = [c for c, _ in res]
        ems = [e for _, e in res]
        summary[s] = (cer, em, percentile(cers[s], .025), percentile(cers[s], .975),
                      percentile(ems, .025), percentile(ems, .975))
    return summary, cers


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="pred_<dataset>.csv written by eval/evaluate.py")
    ap.add_argument("--systems", nargs="*", help="default: every system in the file")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refs", help="JSONL with current references (tgt matched by src), e.g. data/real_test.jsonl")
    args = ap.parse_args(argv)

    systems, stats = load(args.csv, args.refs)
    systems = args.systems or systems
    summary, cers = bootstrap(stats, systems, args.n, args.seed)

    print(f"### {Path(args.csv).name} (n={len(stats[systems[0]])}, {args.n} paired resamples)\n")
    print("| System | CER↓ | 95% CI | EM↑ | 95% CI |\n|---|---:|---|---:|---|")
    for s in systems:
        cer, em, clo, chi, elo, ehi = summary[s]
        print(f"| {s} | {100 * cer:.2f} | {100 * clo:.2f}–{100 * chi:.2f} | {100 * em:.2f} | "
              f"{100 * elo:.2f}–{100 * ehi:.2f} |")
    print("\n| A vs B | CER(A) − CER(B) | 95% CI | A better in |\n|---|---:|---|---:|")
    for a, b in combinations(systems, 2):
        diffs = [x - y for x, y in zip(cers[a], cers[b])]
        d = summary[a][0] - summary[b][0]
        better = sum(x < 0 for x in diffs) / len(diffs)
        print(f"| {a} vs {b} | {100 * d:+.2f} | {100 * percentile(diffs, .025):+.2f}–"
              f"{100 * percentile(diffs, .975):+.2f} | {100 * better:.1f}% |")
    print("\nAn interval for the difference that does not include 0 means the gap is unlikely to be chance.")


if __name__ == "__main__":
    main()
