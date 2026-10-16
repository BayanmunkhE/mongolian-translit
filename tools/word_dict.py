"""Keyboard dictionary cache: the most frequent Latin tokens -> Cyrillic candidates from the teacher's beam.

  python tools/word_dict.py --top 50000 --out data/keyboard/word_dict.json

Frequency comes from the Latin side of the chat pairs, real_train and the v4 synthetic data (never real_dev /
real_test / chat_test). Each word is converted on its own (no context) with beam 4 and up to 3 candidates kept
with their relative probabilities, so the keyboard can answer the common words instantly and fall back to the
model for the rest. Entries whose top candidate equals the input (English words) are kept too: they tell the
keyboard to leave the word alone.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")


def count_words(paths, limit_per_file):
    counts = Counter()
    for path, lim in zip(paths, limit_per_file):
        with open(ROOT / path, encoding="utf-8-sig") as f:
            for i, ln in enumerate(f):
                if lim and i >= lim:
                    break
                if ln.strip():
                    counts.update(w.lower() for w in TOKEN.findall(json.loads(ln)["src"]))
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/byt5-small-chat2")
    ap.add_argument("--top", type=int, default=50000)
    ap.add_argument("--n", type=int, default=3, help="candidates per word")
    ap.add_argument("--out", default="data/keyboard/word_dict.json")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args(argv)

    counts = count_words(["data/chat/train.jsonl", "data/real_train.jsonl", "data/v4/train.jsonl"], [0, 0, 400000])
    words = [w for w, _ in counts.most_common(args.top)]
    print(f"{len(counts)} distinct Latin tokens; converting the top {len(words)}", flush=True)

    import torch
    from translit.model import Seq2SeqTransliterator
    tr = Seq2SeqTransliterator(args.model, num_beams=4)
    out = {}
    for s in range(0, len(words), args.batch):
        batch = words[s:s + args.batch]
        enc = tr.tokenizer(batch, return_tensors="pt", padding=True).to(tr.device)
        with torch.inference_mode():
            gen = tr.model.generate(**enc, num_beams=4, num_return_sequences=args.n, max_new_tokens=64,
                                    early_stopping=True, output_scores=True, return_dict_in_generate=True)
        texts = tr.tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
        lengths = (gen.sequences != tr.tokenizer.pad_token_id).sum(1).float()
        scores = (gen.sequences_scores.float() * lengths).view(len(batch), args.n)
        probs = torch.softmax(scores, dim=1).tolist()
        for i, w in enumerate(batch):
            cands = {}
            for j in range(args.n):
                t = texts[i * args.n + j].strip()
                key = t.lower()
                if t and key not in cands:
                    cands[key] = [t, round(probs[i][j], 3)]
            out[w] = {"count": counts[w], "candidates": sorted(cands.values(), key=lambda c: -c[1])}
        if (s // args.batch) % 20 == 0:
            print(f"{s + len(batch)}/{len(words)}", flush=True)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} entries to {path} ({path.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
