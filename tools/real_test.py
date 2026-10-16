#!/usr/bin/env python3
"""Tooling for the hand-made real test set (data/real_test.jsonl).

This set is ONLY for evaluation: generate_pairs.py excludes its sentences from
all synthetic splits and train_byt5.py refuses to train on it.

Workflows
---------
1) Spreadsheet (recommended for 150-200 items):
     python tools/real_test.py template raw_messages.txt          # one Latin message per line
     -> data/real_test_todo.csv  (src, tgt pre-filled with a *baseline* guess, source, note)
     edit the tgt column in Excel / Google Sheets, save as CSV (UTF-8)
     python tools/real_test.py import data/real_test_todo.csv

2) Interactive, one at a time:
     python tools/real_test.py add

3) Check the set (validation, stats, leakage vs. synthetic data):
     python tools/real_test.py check

4) Correct existing references (CSV with src + new tgt; matched by src, backup written first):
     python tools/real_test.py edit corrections.csv --dry-run

Pre-fills come from a rule baseline, never from the neural model, so the
references cannot be biased toward the system being evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import get_baseline  # noqa: E402
from generate_pairs import normalize_cyrillic, sentence_key  # noqa: E402

REAL = ROOT / "data" / "real_test.jsonl"
CYR_RE = re.compile(r"[Ѐ-ӿ]")
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


# ---------------------------------------------------------------------------
# I/O + validation
# ---------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def validate(src: str, tgt: str) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one pair."""
    errors, warnings = [], []
    if not src:
        errors.append("empty src")
    if not tgt:
        errors.append("empty tgt")
    if errors:
        return errors, warnings
    if CYR_RE.search(src):
        errors.append("src contains Cyrillic — src must be the Latin-script message as written")
    if not CYR_RE.search(tgt):
        errors.append("tgt has no Cyrillic")
    src_words = {w.lower() for w in LATIN_WORD_RE.findall(src)}
    for w in LATIN_WORD_RE.findall(tgt):
        if w.lower() not in src_words:
            warnings.append(f"Latin word {w!r} in tgt does not appear in src")
    if len(tgt) > 3 * len(src) + 10 or len(src) > 3 * len(tgt) + 10:
        warnings.append("src/tgt lengths look mismatched")
    if len(src.encode()) > 250:
        warnings.append("src is long (>250 bytes); the model will split it into segments")
    return errors, warnings


def synthetic_keys() -> set[str]:
    keys = set()
    for name in ("train", "valid", "test"):
        p = ROOT / "data" / f"{name}.jsonl"
        if p.exists():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    keys.add(sentence_key(json.loads(line)["tgt"]))
    return keys


def try_add(items: list[dict], existing_src: set[str], src: str, tgt: str, extra: dict) -> bool:
    src, tgt = clean(src), normalize_cyrillic(clean(tgt))
    errors, warnings = validate(src, tgt)
    if src.lower() in existing_src:
        errors.append("duplicate src")
    for w in warnings:
        print(f"   warning: {w}")
    if errors:
        print(f"   SKIPPED ({'; '.join(errors)}): {src!r}")
        return False
    item = {"src": src, "tgt": tgt}
    item.update({k: v for k, v in extra.items() if v})
    items.append(item)
    existing_src.add(src.lower())
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_template(args):
    sugg = get_baseline(args.suggest) if args.suggest != "none" else None
    lines = [clean(l) for l in Path(args.raw).read_text(encoding="utf-8-sig").splitlines()]
    lines = [l for l in lines if l]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel opens UTF-8 correctly
        w = csv.writer(f)
        w.writerow(["src", "tgt", "source", "note"])
        for l in lines:
            w.writerow([l, sugg(l) if sugg else "", args.source, ""])
    print(f"Wrote {len(lines)} rows to {out}. Fix every tgt cell (it is only a {args.suggest} "
          f"guess), then: python tools/real_test.py import {out}")


def cmd_import(args):
    existing = load(REAL)
    existing_src = {e["src"].lower() for e in existing}
    new = []
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 2):
            print(f"row {i}:", end="")
            ok = try_add(new, existing_src, row.get("src", ""), row.get("tgt", ""),
                         {"source": row.get("source", ""), "note": row.get("note", "")})
            print(" ok" if ok else "")
    if args.dry_run:
        print(f"\n[dry run] would add {len(new)} items")
        return
    append(REAL, new)
    print(f"\nAdded {len(new)} items -> {REAL} (now {len(existing) + len(new)})")


def cmd_add(args):
    sugg = get_baseline(args.suggest) if args.suggest != "none" else None
    existing = load(REAL)
    existing_src = {e["src"].lower() for e in existing}
    print(f"{len(existing)} items in {REAL}.  Commands: :q quit, :s skip.\n"
          "Type the message exactly as it was written in Latin script (keep typos!).")
    while True:
        src = input("\nLatin  > ").strip()
        if src in (":q", ""):
            break
        hint = sugg(src) if sugg else ""
        if hint:
            print(f"  {args.suggest} guess: {hint}")
        tgt = input("Cyrillic (Enter = accept guess) > ").strip() or hint
        if tgt == ":s":
            continue
        new = []
        if try_add(new, existing_src, src, tgt, {"source": args.source}):
            append(REAL, new)
            print(f"  saved ({len(existing_src)} total)")


def cmd_edit(args):
    items = load(REAL)
    by_src = {it["src"].lower(): it for it in items}
    changed = 0
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 2):
            src, new = clean(row.get("src", "")), normalize_cyrillic(clean(row.get("tgt", "")))
            item = by_src.get(src.lower())
            if item is None:
                print(f"row {i}: SKIPPED (src not in the real test): {src!r}")
                continue
            errors, warnings = validate(item["src"], new)
            for w in warnings:
                print(f"row {i}: warning: {w}")
            if errors:
                print(f"row {i}: SKIPPED ({'; '.join(errors)}): {src!r}")
                continue
            if new != item["tgt"]:
                print(f"row {i}: {src}\n   old: {item['tgt']}\n   new: {new}")
                item["tgt"] = new
                changed += 1
    if args.dry_run:
        print(f"\n[dry run] would change {changed} references")
        return
    backup = REAL.with_name(REAL.name + ".bak")
    shutil.copyfile(REAL, backup)
    with REAL.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\nChanged {changed} references in {REAL} (previous version: {backup})")


def cmd_check(args):
    items = load(REAL)
    if not items:
        sys.exit(f"{REAL} is empty or missing")
    n_err = 0
    src_counts = Counter(i["src"].lower() for i in items)
    for k, it in enumerate(items, 1):
        errors, warnings = validate(it["src"], it["tgt"])
        if src_counts[it["src"].lower()] > 1:
            errors.append("duplicate src")
        if "style" in it:
            warnings.append("real test items should not have a 'style' field")
        for e in errors:
            print(f"line {k}: ERROR {e}: {it['src']!r}")
        for w in warnings:
            print(f"line {k}: warning {w}: {it['src']!r}")
        n_err += bool(errors)

    syn = synthetic_keys()
    leaked = [it for it in items if sentence_key(it["tgt"]) in syn]
    n_latin = sum(1 for it in items if LATIN_WORD_RE.search(it["tgt"]))
    sources = Counter(it.get("source", "?") for it in items)
    print(f"\n{len(items)} items, {n_err} with errors, {len(leaked)} also present in synthetic data")
    print(f"items with English/Latin words: {n_latin}")
    print(f"avg src length: {sum(len(i['src']) for i in items) / len(items):.1f} chars")
    print(f"sources: {dict(sources)}")
    for it in leaked[:10]:
        print(f"  LEAK: {it['tgt']}  -> re-run generate_pairs.py (it excludes real_test sentences)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="raw Latin messages (txt) -> CSV to annotate")
    t.add_argument("raw")
    t.add_argument("--out", default=str(ROOT / "data" / "real_test_todo.csv"))
    t.add_argument("--suggest", choices=["naive", "mns5217", "none"], default="naive")
    t.add_argument("--source", default="", help="e.g. messenger, facebook, sms")

    i = sub.add_parser("import", help="append an annotated CSV to real_test.jsonl")
    i.add_argument("csv")
    i.add_argument("--dry-run", action="store_true")

    a = sub.add_parser("add", help="interactive entry")
    a.add_argument("--suggest", choices=["naive", "mns5217", "none"], default="naive")
    a.add_argument("--source", default="")

    e = sub.add_parser("edit", help="correct references of existing items (CSV: src, tgt)")
    e.add_argument("csv")
    e.add_argument("--dry-run", action="store_true")

    sub.add_parser("check", help="validate + leakage check")
    args = ap.parse_args(argv)
    {"template": cmd_template, "import": cmd_import, "add": cmd_add, "edit": cmd_edit,
     "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
