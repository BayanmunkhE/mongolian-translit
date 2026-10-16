#!/usr/bin/env python3
"""Downstream test: does transliteration make Latin-script messages searchable in a Cyrillic corpus?

Each real message is used as a search query against Mongolian Wikipedia (rag/corpus.py) in several
versions: the raw Latin text, the output of each system, and the hand-made Cyrillic reference (gold).
The gold query's top-k passages are the target; for every version we report
    overlap@k  share of the gold top-k that the version's top-k also contains
    top1       % of queries whose first passage equals the gold query's first passage
Queries whose gold version retrieves nothing are skipped. 95% intervals come from a paired bootstrap
over queries. Nothing is tuned here, so running it on the real test is reporting only.

    python eval/retrieval_eval.py
    python eval/retrieval_eval.py --retrievers bm25 --datasets real_dev
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import get_baseline  # noqa: E402
from eval.bootstrap import percentile  # noqa: E402
from rag.retriever import RETRIEVERS, load_passages  # noqa: E402


def load_queries(dataset: str, pred_dirs: list[str], systems: list[str], baselines: list[str]) -> list[dict]:
    """One dict per message: id, latin, gold and one entry per system (model preds read from evaluate.py CSVs)."""
    rows, preds = None, {}
    for d in pred_dirs:
        with open(ROOT / d / f"pred_{dataset}.csv", encoding="utf-8-sig", newline="") as f:
            table = list(csv.DictReader(f))
        rows = rows or table
        for r in table:
            for s in systems:
                if f"{s}_pred" in r:
                    preds[(r["id"], s)] = r[f"{s}_pred"]
    missing = [s for s in systems if (rows[0]["id"], s) not in preds]
    if missing:
        sys.exit(f"no predictions for {missing} in {pred_dirs}")
    queries = [{"id": r["id"], "latin": r["src"], "gold": r["tgt"]} for r in rows]
    for name in baselines:
        for q, out in zip(queries, get_baseline(name).batch([q["latin"] for q in queries])):
            q[name] = out
    for q in queries:
        for s in systems:
            q[s] = preds[(q["id"], s)]
    return queries


def score(results: dict[str, list], variants: list[str]) -> dict[str, list[tuple[float, int]]]:
    """Per variant, one (overlap, top1) pair per query that has gold results."""
    out = {v: [] for v in variants}
    for i, gold in enumerate(results["gold"]):
        if not gold:
            continue
        g = [d for d, _ in gold]
        for v in variants:
            got = [d for d, _ in results[v][i]]
            out[v].append((len(set(g) & set(got)) / len(g), int(bool(got) and got[0] == g[0])))
    return out


def bootstrap(stats: dict, variants: list[str], n: int, seed: int):
    size = len(next(iter(stats.values())))
    rng = random.Random(seed)
    samples = [[rng.randrange(size) for _ in range(size)] for _ in range(n)]
    summary, overlaps = {}, {}
    for v in variants:
        s = stats[v]
        overlaps[v] = [sum(s[i][0] for i in idx) / size for idx in samples]
        top1 = [sum(s[i][1] for i in idx) / size for idx in samples]
        summary[v] = (sum(o for o, _ in s) / size, percentile(overlaps[v], .025), percentile(overlaps[v], .975),
                      sum(t for _, t in s) / size, percentile(top1, .025), percentile(top1, .975))
    return summary, overlaps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=["real_dev", "real_test"])
    ap.add_argument("--pred-dir", nargs="*", default=["results/v4b", "results/v4"],
                    help="dirs with evaluate.py pred_<dataset>.csv files (first one gives the rows)")
    ap.add_argument("--systems", nargs="*", default=["byt5-v3", "small-v4-real", "base-v4-real"])
    ap.add_argument("--baselines", nargs="*", default=["naive", "mns5217"])
    ap.add_argument("--retrievers", nargs="*", default=list(RETRIEVERS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/rag")
    args = ap.parse_args(argv)

    passages = load_passages()
    retrievers = {name: RETRIEVERS[name](passages) for name in args.retrievers}
    variants = ["latin", *args.baselines, *args.systems]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for ds in args.datasets:
        queries = load_queries(ds, args.pred_dir, args.systems, args.baselines)
        per_query = [{"id": q["id"], "latin": q["latin"], "gold": q["gold"]} for q in queries]
        for rname, retriever in retrievers.items():
            results = {v: retriever.search([q[v] for q in queries], k=args.k) for v in ["gold", *variants]}
            stats = score(results, variants)
            kept = [i for i, g in enumerate(results["gold"]) if g]
            for v in variants:
                for i, (o, _) in zip(kept, stats[v]):
                    per_query[i][f"{rname}:{v}"] = f"{o:.2f}"
            summary, overlaps = bootstrap(stats, variants, args.n, args.seed)
            report += [f"### {ds}, {rname} (n={len(kept)} of {len(queries)} queries with gold results, k={args.k})",
                       "", "| Query | overlap@k ↑ | 95% CI | top-1 agreement ↑ | 95% CI |", "|---|---:|---|---:|---|"]
            for v in variants:
                o, olo, ohi, t, tlo, thi = summary[v]
                report.append(f"| {v} | {100 * o:.1f} | {100 * olo:.1f}–{100 * ohi:.1f} | "
                              f"{100 * t:.1f} | {100 * tlo:.1f}–{100 * thi:.1f} |")
            report += ["", "| A vs B | overlap(A) − overlap(B) | 95% CI |", "|---|---:|---|"]
            for a, b in combinations([*args.baselines[-1:], *args.systems], 2):
                diffs = sorted(x - y for x, y in zip(overlaps[a], overlaps[b]))
                report.append(f"| {a} vs {b} | {100 * (summary[a][0] - summary[b][0]):+.1f} | "
                              f"{100 * percentile(diffs, .025):+.1f}–{100 * percentile(diffs, .975):+.1f} |")
            report.append("")
        with open(out_dir / f"retrieval_{ds}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_query[0]) +
                               [f"{r}:{v}" for r in retrievers for v in variants if f"{r}:{v}" not in per_query[0]])
            w.writeheader()
            w.writerows(per_query)
    report.append("overlap@k: share of the gold (hand-made Cyrillic) query's top-k passages that the query version "
                  "also retrieves. Gold vs itself is 100 by definition.")
    text = "\n".join(report)
    (out_dir / "retrieval.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
