"""Keyboard question: does converting a word together with the two previous Latin words beat converting it alone?

  python tools/context_eval.py --model models/byt5-pruned-4e2d-1024-ft2

For every message whose Latin and Cyrillic references have the same number of words (so word i aligns with word i),
each word is converted (a) alone and (b) as the last word of "prev2 prev1 word", taking the tail of the output
after the conversion of "prev2 prev1" (the keyboard already knows that prefix's conversion). Reports word-level
accuracy (case-insensitive) for both, per test set. Words are compared with the reference word in the same position.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from translit.model import Seq2SeqTransliterator  # noqa: E402


def load(path):
    rows = []
    with open(ROOT / path, encoding="utf-8-sig") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                s, t = d["src"].split(), d["tgt"].split()
                if len(s) == len(t) and len(s) >= 1:
                    rows.append((s, t))
    return rows


def norm(w):
    return w.lower().strip(".,!?:;\"'()")


def tail_after_prefix(full, prefix_conv, n_words):
    """The part of `full` that follows the known conversion of the prefix; falls back to the last n words."""
    if prefix_conv and full.lower().startswith(prefix_conv.lower()):
        rest = full[len(prefix_conv):].strip()
        if rest:
            return rest
    words = full.split()
    return " ".join(words[-n_words:]) if words else full


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/byt5-pruned-4e2d-1024-ft2")
    ap.add_argument("--sets", nargs="*", default=["chat_test=data/chat_test.jsonl", "real_test=data/real_test.jsonl"])
    ap.add_argument("--context", type=int, default=2)
    args = ap.parse_args(argv)
    tr = Seq2SeqTransliterator(args.model, num_beams=1)
    conv = tr.transliterate
    lines = [f"# Context vs standalone word conversion: {args.model}", ""]
    for spec in args.sets:
        name, path = spec.split("=", 1)
        rows = load(path)
        n = alone_ok = ctx_ok = both_wrong = 0
        fixes, breaks = [], []
        t0 = time.time()
        for s, t in rows:
            for i, w in enumerate(s):
                ref = norm(t[i])
                a = norm(conv(w))
                prefix = s[max(0, i - args.context):i]
                if prefix:
                    full = conv(" ".join(prefix + [w]))
                    c = norm(tail_after_prefix(full, conv(" ".join(prefix)), 1))
                else:
                    c = a
                n += 1
                alone_ok += a == ref
                ctx_ok += c == ref
                both_wrong += a != ref and c != ref
                if a != ref and c == ref and len(fixes) < 8:
                    fixes.append(f"{' '.join(prefix)} [{w}] → alone '{a}', context '{c}'")
                if a == ref and c != ref and len(breaks) < 8:
                    breaks.append(f"{' '.join(prefix)} [{w}] → alone '{a}', context '{c}'")
        lines += [f"## {name}: {len(rows)} aligned messages, {n} words ({time.time() - t0:.0f}s)", "",
                  f"- word accuracy alone: **{100 * alone_ok / n:.1f}%**",
                  f"- word accuracy with {args.context}-word context: **{100 * ctx_ok / n:.1f}%**",
                  f"- both wrong: {100 * both_wrong / n:.1f}%", "", "Fixed by context:", *[f"- {x}" for x in fixes], "",
                  "Broken by context:", *[f"- {x}" for x in breaks], ""]
        print("\n".join(lines[-14:]), flush=True)
    out = ROOT / "results" / "keyboard" / "context_eval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
