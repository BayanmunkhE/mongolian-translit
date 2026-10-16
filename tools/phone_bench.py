"""Step 1 of the on-device plan: how does the current model behave under phone-like constraints?

  python tools/phone_bench.py --model models/byt5-small-chat

Runs the test sets one message at a time (like a keyboard) on CPU and reports, per configuration,
CER / exact match and per-message latency (p50 / p95):
  fp32 beam4   - what the demo serves today
  fp32 greedy  - no beam search (phones decode greedily)
  int8 greedy  - dynamic int8 quantization of the Linear layers (what a mobile runtime would do)
Also prints the weight size on disk and the int8 state size. The numbers calibrate how small the student
model for the phone can be and how much accuracy greedy + int8 cost by themselves.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import corpus_metrics  # noqa: E402
from translit.model import Seq2SeqTransliterator  # noqa: E402


def load(path):
    rows = []
    with open(ROOT / path, encoding="utf-8-sig") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                rows.append((d["src"], d["tgt"]))
    return rows


def run(translit, rows, limit):
    hyps, times = [], []
    for src, _ in rows[:limit] if limit else rows:
        translit._cached.cache_clear()
        t0 = time.perf_counter()
        hyps.append(translit.transliterate(src))
        times.append((time.perf_counter() - t0) * 1000)
    refs = [t for _, t in (rows[:limit] if limit else rows)]
    m = corpus_metrics(refs, hyps)
    times.sort()
    return {"cer": round(100 * m["cer"], 2), "em": round(100 * m["exact"], 1),
            "p50_ms": round(times[len(times) // 2]), "p95_ms": round(times[int(len(times) * 0.95) - 1]),
            "p99_ms": round(times[min(len(times) - 1, int(len(times) * 0.99))]),
            "mean_ms": round(statistics.mean(times))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/byt5-small-chat")
    ap.add_argument("--sets", nargs="*", default=["real_test=data/real_test.jsonl", "chat_test=data/chat_test.jsonl"])
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (a phone has 4-8 slow cores)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/phone/bench.md")
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    sets = {name: load(path) for name, path in (s.split("=", 1) for s in args.sets)}
    weights = sum(p.stat().st_size for p in (ROOT / args.model).glob("*.safetensors")) / 2**20

    configs = [("fp32 beam4", 4, False), ("fp32 greedy", 1, False), ("int8 greedy", 1, True)]
    results, int8_mb = {}, None
    for label, beams, quant in configs:
        tr = Seq2SeqTransliterator(args.model, device="cpu", num_beams=beams)
        if quant:
            # encoder/decoder Linear layers only: int8 on lm_head wrecked the pruned student (real CER 11.5 -> 37.9)
            for stack in ("encoder", "decoder"):
                setattr(tr.model, stack, torch.ao.quantization.quantize_dynamic(
                    getattr(tr.model, stack), {torch.nn.Linear}, dtype=torch.qint8))
            tmp = ROOT / "results" / "phone" / "int8_state.pt"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tr.model.state_dict(), tmp)
            int8_mb = tmp.stat().st_size / 2**20
            os.remove(tmp)
        for name, rows in sets.items():
            results[(label, name)] = run(tr, rows, args.limit)
            print(label, name, results[(label, name)], flush=True)
        del tr

    lines = [f"# Phone-constraint benchmark: {args.model}", "",
             f"CPU {args.threads} threads, one message at a time, beam = decoding. "
             f"Weights on disk: {weights:.0f} MB fp32; int8 Linear layers: {int8_mb:.0f} MB.", ""]
    for name, rows in sets.items():
        lines += [f"### {name} (n={args.limit or len(rows)})", "",
                  "| Config | CER↓ | EM↑ | p50 ms | p95 ms | p99 ms | mean ms |", "|---|---:|---:|---:|---:|---:|---:|"]
        for label, _, _ in configs:
            r = results[(label, name)]
            lines.append(f"| {label} | {r['cer']} | {r['em']} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | {r['mean_ms']} |")
        lines.append("")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
