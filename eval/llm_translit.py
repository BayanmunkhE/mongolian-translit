#!/usr/bin/env python3
"""LLM baseline: an LLM converts Latin Mongolian to Cyrillic, with and without retrieved examples.

Providers: Claude (Anthropic API) or a local open model served by Ollama (free, offline):
    python eval/llm_translit.py --provider ollama --model gemma3:12b

Three prompting conditions per model:
    zero   instructions only (the annotation rules)
    rand5  + 5 random pairs from data/real_train.jsonl
    rag5   + the 5 real_train pairs most similar to the input (BM25 over character 3-grams of the Latin)
rand5 vs rag5 isolates what retrieval adds. Examples only ever come from real_train; real_dev/real_test
are never shown to the model as examples. Iterate on the prompt with real_dev only; real_test is reporting.

    python eval/llm_translit.py --dry-run                      # prompts, retrieved examples, cost estimate; no API
    python eval/llm_translit.py --limit 20 --datasets real_dev # cheap pilot
    python eval/llm_translit.py                                # full run -> results/llm/
    python eval/bootstrap.py results/llm/pred_real_test.csv

Responses are cached in results/llm/cache.jsonl, so an interrupted run resumes without paying twice.
Claude needs ANTHROPIC_API_KEY (or an `ant auth login` profile); Ollama needs `ollama serve` and the model pulled.
Every system's predictions are merged into results/llm/pred_<dataset>.csv across runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.evaluate import load_jsonl, markdown_table  # noqa: E402
from eval.metrics import cer, corpus_metrics  # noqa: E402

SYSTEM = """You convert informal Mongolian written in Latin letters (as typed in chats and YouTube comments) \
into standard Mongolian Cyrillic.

Rules:
- Write what the writer meant, in correct Mongolian spelling: ө/о, ү/у, х, и/й/ь, long vowels \
(margash → маргааш). Fix the writer's spelling mistakes.
- Expand chat abbreviations: bn → байна, bh → байх, shd → шүү дээ.
- Keep English words, numbers, emoji, links and @names exactly as written (like, comment, 2023, <3).
  A Mongolian suffix on a number or an English word is joined with a hyphen (2iin → 2-ын, motived → motive-д).
- Keep capitalization as in the source: a message that starts in lowercase stays lowercase.
- Do not add punctuation the writer did not type; keep the punctuation they did type.
- Do not translate, rephrase, add or drop words.

Put only the converted text in the "cyrillic" field."""

SCHEMA = {"type": "object", "properties": {"cyrillic": {"type": "string"}},
          "required": ["cyrillic"], "additionalProperties": False}
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
          "claude-fable-5-1": (10.0, 50.0)}  # $ per 1M input / output tokens


class ExampleRetriever:
    """Nearest real_train pairs by BM25 over character 3-grams of the lowercased Latin text."""

    def __init__(self, pool: list[dict]):
        import bm25s
        self.pool = pool
        self.index = bm25s.BM25()
        self.index.index([self.grams(p["src"]) for p in pool], show_progress=False)
        self.vocab = self.index.vocab_dict

    @staticmethod
    def grams(s: str) -> list[str]:
        s = f" {' '.join(s.lower().split())} "
        return [s[i:i + 3] for i in range(len(s) - 2)]

    def __call__(self, src: str, k: int) -> list[dict]:
        tokens = [g for g in self.grams(src) if g in self.vocab]
        if not tokens:
            return []
        docs, scores = self.index.retrieve([tokens], k=k, show_progress=False)
        # most similar last, right before the input
        return [self.pool[int(d)] for d, s in zip(docs[0], scores[0]) if s > 0][::-1]


def user_prompt(src: str, examples: list[dict]) -> str:
    parts = []
    if examples:
        parts.append("Examples of the expected conversion:\n")
        parts += [f"Latin: {e['src']}\nCyrillic: {e['tgt']}\n" for e in examples]
    parts.append(f"Convert this message:\n<message>\n{src}\n</message>")
    return "\n".join(parts)


def build_prompts(rows: list[dict], cond: str, pool: list[dict], retriever: ExampleRetriever, k: int) -> list[str]:
    prompts = []
    for r in rows:
        if cond == "zero":
            ex = []
        elif cond == f"rand{k}":
            ex = random.Random(r["src"]).sample(pool, k)  # seeded by the input: same examples on every run
        else:
            ex = retriever(r["src"], k)
        prompts.append(user_prompt(r["src"], ex))
    return prompts


class Cache:
    def __init__(self, path: Path):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    self.data[d["key"]] = d

    def put(self, d: dict):
        with self.lock:
            self.data[d["key"]] = d
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


def parse(text: str, out: dict) -> dict:
    try:
        out["pred"] = json.loads(text)["cyrillic"].strip()
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        out["stop_reason"] = f"unparsed:{out['stop_reason']}"
    return out


def call_ollama(args, prompt: str) -> dict:
    """Local model via Ollama's chat API; `format` constrains the reply to SCHEMA, temperature 0."""
    body = {"model": args.model, "stream": False, "format": SCHEMA, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{args.ollama_url}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    out = {"served_by": d.get("model", args.model), "stop_reason": d.get("done_reason", "stop"),
           "input_tokens": d.get("prompt_eval_count", 0), "output_tokens": d.get("eval_count", 0), "pred": ""}
    return parse(d.get("message", {}).get("content", ""), out)


def call(client, args, prompt: str) -> dict:
    if args.provider == "ollama":
        return call_ollama(args, prompt)
    kwargs = dict(model=args.model, max_tokens=8000, system=SYSTEM,
                  messages=[{"role": "user", "content": prompt}],
                  output_config={"effort": args.effort, "format": {"type": "json_schema", "schema": SCHEMA}})
    if args.model in FALLBACK_MODELS:  # a policy decline is re-run on a fallback model; recorded in served_by
        resp = client.beta.messages.create(**kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    else:
        resp = client.messages.create(**kwargs)
    out = {"served_by": resp.model, "stop_reason": resp.stop_reason,
           "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens, "pred": ""}
    if resp.stop_reason == "refusal":
        return out
    return parse(next((b.text for b in resp.content if b.type == "text"), ""), out)


def merge_previous(path: Path, rows: list[dict], preds: dict[str, list[str]]):
    """Keep other systems' columns from an earlier run on the same rows (e.g. Claude after Ollama)."""
    if not path.exists():
        return
    with open(path, encoding="utf-8-sig", newline="") as f:
        old = list(csv.DictReader(f))
    if [r["src"] for r in old] != [r["src"] for r in rows]:
        return  # a pilot (--limit) or other rows: nothing to merge
    for col in old[0]:
        s = col.removesuffix("_pred")
        if col.endswith("_pred") and s not in preds:
            preds[s] = [r[col] for r in old]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "ollama"])
    ap.add_argument("--model", default="claude-opus-5-5", help="e.g. claude-opus-5-5, or gemma3:12b with ollama")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude only")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--label", help="system name prefix (default: model id without 'claude-', ':' as '-')")
    ap.add_argument("--conditions", nargs="*", default=["zero", "rand5", "rag5"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--datasets", nargs="*", default=["real_dev", "real_test"])
    ap.add_argument("--pool", default="data/real_train.jsonl")
    ap.add_argument("--compare-dir", default="results/v4b", help="evaluate.py output with ByT5 predictions")
    ap.add_argument("--compare", nargs="*", default=["small-v4-real", "base-v4-real"])
    ap.add_argument("--limit", type=int, default=0, help="first N messages per dataset (pilot)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="results/llm")
    ap.add_argument("--dry-run", action="store_true", help="print prompts and a cost estimate; no API calls")
    args = ap.parse_args(argv)
    for p in (args.pool,):
        if "real_test" in Path(p).name or "real_dev" in Path(p).name:
            sys.exit("The example pool must not contain evaluation data.")
    args.conditions = [c.replace("5", str(args.k)) if c != "zero" else c for c in args.conditions]
    label = args.label or args.model.removeprefix("claude-").replace(":", "-")
    if args.provider == "ollama":
        args.effort = "-"  # not an Ollama setting; keeps cache keys and the report honest

    pool = load_jsonl(ROOT / args.pool)
    retriever = ExampleRetriever(pool)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(out_dir / "cache.jsonl")
    client = None
    if not args.dry_run and args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()

    md = ["# LLM transliteration", "", f"Last run: {args.provider} {args.model}, effort {args.effort}, k={args.k}. "
          "Tables include systems merged from earlier runs on the same rows.", ""]
    totals = [0, 0]
    for ds in args.datasets:
        rows = load_jsonl(ROOT / "data" / f"{ds}.jsonl", args.limit)
        with open(ROOT / args.compare_dir / f"pred_{ds}.csv", encoding="utf-8-sig", newline="") as f:
            compare = {r["src"]: r for r in csv.DictReader(f)}
        preds = {s: [compare[r["src"]][f"{s}_pred"] for r in rows] for s in args.compare}
        for cond in args.conditions:
            prompts = build_prompts(rows, cond, pool, retriever, args.k)
            keys = [hashlib.sha1(f"{args.provider}|{args.model}|{args.effort}|{SYSTEM}|{p}".encode()).hexdigest()
                    for p in prompts]
            todo = [(k, p) for k, p in dict(zip(keys, prompts)).items() if k not in cache.data]
            if args.dry_run:
                chars = sum(len(SYSTEM) + len(p) for p in prompts)
                print(f"--- {ds} / {cond}: {len(prompts)} prompts, {len(todo)} not cached, ~{chars // 2:,} input "
                      f"tokens (rough: ~2 chars per token for this mix)\n{prompts[0]}\n")
                totals[0] += chars // 2
                continue

            def run(item):
                key, prompt = item
                try:
                    cache.put({"key": key, **call(client, args, prompt)})
                except Exception as e:  # noqa: BLE001 - keep the batch going; uncached items retry next run
                    print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)

            with ThreadPoolExecutor(args.workers) as ex:
                list(ex.map(run, todo))
            got = [cache.data.get(k) for k in keys]
            missing = sum(g is None for g in got)
            if missing:
                print(f"[warn] {ds}/{cond}: {missing} calls failed; re-run to retry them", file=sys.stderr)
            preds[f"{label}-{cond}"] = [g["pred"] if g else "" for g in got]
            done = [g for g in got if g]
            totals[0] += sum(g["input_tokens"] for g in done)
            totals[1] += sum(g["output_tokens"] for g in done)
            odd = [g for g in done if g["stop_reason"] not in ("end_turn", "stop")
                   or (args.provider == "anthropic" and g["served_by"] != args.model)]
            if odd:
                print(f"[note] {ds}/{cond}: {len(odd)} responses refused, unparsed or served by a fallback "
                      f"model: {sorted({(g['stop_reason'], g['served_by']) for g in odd})}", file=sys.stderr)
        if args.dry_run:
            continue
        refs = [r["tgt"] for r in rows]
        merge_previous(out_dir / f"pred_{ds}.csv", rows, preds)
        md.append(markdown_table(ds, {s: corpus_metrics(refs, p) for s, p in preds.items()}))
        with open(out_dir / f"pred_{ds}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "src", "tgt"] + [c for s in preds for c in (f"{s}_pred", f"{s}_cer")])
            for i, r in enumerate(rows):
                w.writerow([i, r["src"], r["tgt"]] +
                           [c for s in preds for c in (preds[s][i], f"{cer(r['tgt'], preds[s][i]):.4f}")])

    price = PRICES.get(args.model)
    if args.dry_run:
        if price:
            print(f"Estimated input cost: ${totals[0] * price[0] / 1e6:.2f} (+ output and thinking tokens; "
                  f"run --limit 20 first to measure them)")
        return
    if price:
        md.append(f"Tokens: {totals[0]:,} in / {totals[1]:,} out, about "
                  f"${(totals[0] * price[0] + totals[1] * price[1]) / 1e6:.2f} (cached responses included).")
    report = "\n".join(md)
    (out_dir / "results.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
