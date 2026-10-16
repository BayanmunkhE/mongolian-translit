#!/usr/bin/env python3
"""Run every system on the synthetic and real test sets and report metrics.

    python eval/evaluate.py                                   # baselines only
    python eval/evaluate.py --model models/byt5-mn            # + fine-tuned model
    python eval/evaluate.py --model models/byt5-mn --model models/mt5-mn

Writes:
    results/results.md          markdown tables (also printed)
    results/metrics.json        all numbers
    results/pred_<dataset>.csv  per-example predictions of every system
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import BASELINES, get_baseline  # noqa: E402
from eval.metrics import AMBIGUOUS_GROUPS, cer, corpus_metrics  # noqa: E402

COLUMNS = [("cer", "CER↓"), ("wer", "WER↓"), ("exact", "EM↑"),
           ("cer_ci", "CER (ci)↓"), ("exact_ci", "EM (ci)↑"), ("ambig_acc", "Ambig↑")] + \
          [(f"acc[{g}]", g) for g in AMBIGUOUS_GROUPS] + [("eng_preserved", "Eng↑"), ("chrf", "chrF↑"), ("bleu", "BLEU↑")]


def load_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def fmt(v) -> str:
    if isinstance(v, float):
        return "–" if math.isnan(v) else f"{100 * v:.2f}"
    return str(v)


def markdown_table(dataset: str, results: dict[str, dict]) -> str:
    n = next(iter(results.values()))["n"]
    lines = [f"### {dataset} (n={n})", "",
             "| System | " + " | ".join(h for _, h in COLUMNS) + " |",
             "|---|" + "---:|" * len(COLUMNS)]
    for sys_name, m in results.items():
        lines.append(f"| {sys_name} | " + " | ".join(fmt(m[k]) for k, _ in COLUMNS) + " |")
    eng = next(iter(results.values()))["eng_total"]
    lines += ["", f"All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous "
                  f"letters ({', '.join(AMBIGUOUS_GROUPS)}). Eng = Latin words preserved "
                  f"({eng} in references).", ""]
    return "\n".join(lines)


def build_systems(args):
    systems = {}
    for name in args.baselines:
        systems[name] = get_baseline(name).batch
    for model_dir in args.model:
        from translit.model import Seq2SeqTransliterator
        tr = Seq2SeqTransliterator(model_dir, num_beams=args.num_beams)
        label = args.model_name.pop(0) if args.model_name else tr.name
        systems[label] = lambda texts, tr=tr: tr.batch(texts, batch_size=args.batch_size)
    return systems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default="data/test.jsonl", help="synthetic test set")
    ap.add_argument("--real", default="data/real_test.jsonl", help="hand-made real test set")
    ap.add_argument("--extra-test", nargs="*", default=[], help="more name=path test sets")
    ap.add_argument("--baselines", nargs="*", default=list(BASELINES))
    ap.add_argument("--model", action="append", default=[], help="fine-tuned model dir (repeatable)")
    ap.add_argument("--model-name", action="append", default=[], help="label per --model")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N examples")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)

    datasets = {"synthetic_test": Path(args.test), "real_test": Path(args.real)}
    for spec in args.extra_test:
        name, path = spec.split("=", 1)
        datasets[name] = Path(path)

    systems = build_systems(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics, md = {}, ["# Evaluation results", ""]

    for ds_name, path in datasets.items():
        if not path.exists():
            print(f"[skip] {ds_name}: {path} not found", file=sys.stderr)
            continue
        rows = load_jsonl(path, args.limit)
        if not rows:
            print(f"[skip] {ds_name}: {path} is empty", file=sys.stderr)
            continue
        srcs, refs = [r["src"] for r in rows], [r["tgt"] for r in rows]
        preds, ds_metrics = {}, {}
        for sys_name, fn in systems.items():
            t0 = time.time()
            preds[sys_name] = fn(srcs)
            secs = time.time() - t0
            ds_metrics[sys_name] = corpus_metrics(refs, preds[sys_name])
            ds_metrics[sys_name]["seconds"] = round(secs, 2)
            print(f"{ds_name:>15} | {sys_name:>12} | CER {100 * ds_metrics[sys_name]['cer']:6.2f} "
                  f"| EM {100 * ds_metrics[sys_name]['exact']:6.2f} | {secs:.1f}s", file=sys.stderr)
        all_metrics[ds_name] = ds_metrics
        md.append(markdown_table(ds_name, ds_metrics))

        csv_path = out_dir / f"pred_{ds_name}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "src", "tgt"] + [c for s in systems for c in (f"{s}_pred", f"{s}_cer")])
            for i, (src, ref) in enumerate(zip(srcs, refs)):
                row = [i, src, ref]
                for s in systems:
                    row += [preds[s][i], f"{cer(ref, preds[s][i]):.4f}"]
                w.writerow(row)

    report = "\n".join(md)
    print(report)
    (out_dir / "results.md").write_text(report, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\nSaved {out_dir / 'results.md'}, metrics.json and pred_*.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
