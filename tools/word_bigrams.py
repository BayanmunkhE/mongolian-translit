"""Bigram counts for the keyboard dictionary: how often a Cyrillic word follows another, so the suggestion strip
can rank "мин" vs "минь" by the previous committed word.

  python tools/word_bigrams.py --out data/keyboard/bigrams.json

Counts come from the Cyrillic side of the chat pairs and the first --max-synthetic rows of the v4 synthetic data.
Only pairs whose next word is a dictionary candidate (data/keyboard/word_dict.json) are kept, at most --keep
successors per previous word, count >= --min-count.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[\w'-]+", re.UNICODE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict", default="data/keyboard/word_dict.json")
    ap.add_argument("--max-synthetic", type=int, default=400000)
    ap.add_argument("--keep", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--out", default="data/keyboard/bigrams.json")
    args = ap.parse_args(argv)
    cands = set()
    for entry in json.loads((ROOT / args.dict).read_text(encoding="utf-8")).values():
        cands.update(c[0].lower() for c in entry["candidates"])
    pairs = defaultdict(Counter)
    for path, limit in (("data/chat/train.jsonl", 0), ("data/real_train.jsonl", 0), ("data/v4/train.jsonl", args.max_synthetic)):
        with open(ROOT / path, encoding="utf-8-sig") as f:
            for i, ln in enumerate(f):
                if limit and i >= limit:
                    break
                if not ln.strip():
                    continue
                words = [w.lower() for w in WORD.findall(json.loads(ln)["tgt"])]
                for a, b in zip(words, words[1:]):
                    if b in cands:
                        pairs[a][b] += 1
    out = {}
    for prev, nxt in pairs.items():
        top = [(w, c) for w, c in nxt.most_common(args.keep) if c >= args.min_count]
        if top:
            out[prev] = dict(top)
    path = ROOT / args.out
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} previous words, {sum(len(v) for v in out.values())} pairs -> {path} ({path.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
