#!/usr/bin/env python3
"""Print the most frequent error types of one system from a predictions CSV.

    python eval/error_analysis.py results/pred_real_test.csv --system byt5-mn
    python eval/error_analysis.py results/pred_synthetic_test.csv --system mns5217 --top 30
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import align, latin_words, norm  # noqa: E402

VOWELS = set("аэиоуөүеёюяы")
PAIR_CATEGORIES = [
    ("ө ↔ о", set("өо")), ("ү ↔ у", set("үу")), ("ө ↔ ү/у", set("өүу")),
    ("и/й/ь/ъ/ы", set("ийьъы")), ("ш ↔ щ", set("шщ")), ("е ↔ э", set("еэ")),
    ("ц ↔ ч/с", set("цчс")), ("ж ↔ з", set("жз")),
]


def categorize(r: str | None, h: str | None, ref: str, ri: int | None) -> str:
    """Classify one character-level edit (case already folded)."""
    if r is not None and h is not None:
        for name, chars in PAIR_CATEGORIES:
            if r in chars and h in chars:
                return name
        if r.isascii() and r.isalpha():
            return "English word altered"
        if not r.isalnum() or not h.isalnum():
            return "punctuation/space"
        return "other substitution"
    ch = r if r is not None else h
    if ch in VOWELS:
        # missing/extra half of a long vowel or diphthong (аа, өө, ий ...)
        if r is not None and ri is not None and (
                (ri > 0 and ref[ri - 1] in VOWELS) or (ri + 1 < len(ref) and ref[ri + 1] in VOWELS)):
            return "long vowel shortened"
        return "vowel inserted" if r is None else "vowel deleted"
    if ch in "ьъ":
        return "soft/hard sign missing" if r is not None else "soft/hard sign inserted"
    if not ch.isalnum():
        return "punctuation/space"
    return "consonant inserted" if r is None else "consonant deleted"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--system", help="system column prefix (default: last system in the CSV)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--worst", type=int, default=10, help="print N worst examples")
    args = ap.parse_args(argv)

    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    systems = [c[:-5] for c in rows[0] if c.endswith("_pred")]
    system = args.system or systems[-1]
    if system not in systems:
        sys.exit(f"unknown system {system!r}; available: {systems}")
    col = f"{system}_pred"

    cats, subs, dels, ins = Counter(), Counter(), Counter(), Counter()
    word_errs, case_only, n_err_sent = Counter(), 0, 0
    eng_lost = Counter()
    for row in rows:
        ref_raw, hyp_raw = norm(row["tgt"]), norm(row[col])
        if ref_raw == hyp_raw:
            continue
        n_err_sent += 1
        ref, hyp = ref_raw.lower(), hyp_raw.lower()
        if ref == hyp:
            case_only += 1
            cats["case only"] += 1
            continue
        for ri, hi in align(ref, hyp):
            r = ref[ri] if ri is not None else None
            h = hyp[hi] if hi is not None else None
            if r == h:
                continue
            cats[categorize(r, h, ref, ri)] += 1
            if r and h:
                subs[f"{r}→{h}"] += 1
            elif r:
                dels[r] += 1
            else:
                ins[h] += 1
        rw, hw = ref.split(), hyp.split()
        for wi, wj in align(rw, hw):
            if wi is not None and wj is not None and rw[wi] != hw[wj]:
                word_errs[f"{rw[wi]} → {hw[wj]}"] += 1
        hyp_words = latin_words(hyp_raw)
        for w, c in latin_words(ref_raw).items():
            if hyp_words[w] < c:
                eng_lost[w] += c - hyp_words[w]

    def show(title, counter, n=args.top):
        total = sum(counter.values())
        print(f"\n== {title} (total {total}) ==")
        for k, v in counter.most_common(n):
            print(f"  {v:6d}  {100 * v / max(1, total):5.1f}%  {k}")

    print(f"System: {system}   file: {args.csv}")
    print(f"Sentences with errors: {n_err_sent}/{len(rows)}  (case-only: {case_only})")
    show("Error categories (char edits)", cats)
    show("Top substitutions ref→hyp", subs)
    show("Top deleted chars", dels, 10)
    show("Top inserted chars", ins, 10)
    show("Top word errors ref → hyp", word_errs)
    if eng_lost:
        show("English/Latin words not preserved", eng_lost, 10)

    if args.worst:
        print(f"\n== {args.worst} worst examples by CER ==")
        worst = sorted(rows, key=lambda r: -float(r[f"{system}_cer"]))[:args.worst]
        for r in worst:
            print(f"  CER {float(r[f'{system}_cer']):.2f}\n    src: {r['src']}\n"
                  f"    ref: {r['tgt']}\n    hyp: {r[col]}")


if __name__ == "__main__":
    main()
