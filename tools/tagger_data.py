"""Labels for the encoder-only tagger (on-device track 5): each input byte gets the output string it produces.

  python tools/tagger_data.py --train data/student_train.jsonl --max-train 400000

Alignment: minimum-edit alignment of source and target characters (eval.metrics.align). Each target character
is attached to the source character it aligns with; inserted target characters go to the previous source
character; deleted source characters get the empty label. A multi-byte source character (emoji, Cyrillic in mixed
text) carries its label on its first byte, continuation bytes get the empty label. Labels are whole strings
("б", "байна", "", "үү"); the vocabulary keeps the most frequent ones, the rest map to UNK (the keyboard then
falls back to the seq2seq model or copies the source character).

Writes data/tagger/{train,valid}.jsonl ({"src", "labels": [label ids per byte]}) and data/tagger/vocab.json.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import align  # noqa: E402

OUT = ROOT / "data" / "tagger"
EMPTY, UNK = "", "<unk>"


def char_labels(src: str, tgt: str) -> list[str]:
    """One output string per source character."""
    labels = [""] * len(src)
    last = 0
    for i, j in align(src, tgt):
        if i is not None:
            last = i
            if j is not None:
                labels[i] += tgt[j]
        elif j is not None:
            labels[last] += tgt[j]
    return labels


def byte_labels(src: str, tgt: str) -> list[str]:
    """One label per UTF-8 byte of src (label on the first byte of each character)."""
    out = []
    for ch, lab in zip(src, char_labels(src, tgt)):
        n = len(ch.encode("utf-8"))
        out.append(lab)
        out.extend([EMPTY] * (n - 1))
    return out


def read(path, limit=0):
    rows = []
    with open(ROOT / path, encoding="utf-8-sig") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                rows.append((d["src"], d["tgt"]))
                if limit and len(rows) >= limit:
                    break
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/student_train.jsonl")
    ap.add_argument("--valid", default="data/chat_dev_mix.jsonl")
    ap.add_argument("--max-train", type=int, default=400000)
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--max-bytes", type=int, default=200)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    train = read(args.train, args.max_train)
    valid = read(args.valid)
    counts = Counter()
    labelled = []
    for src, tgt in train:
        if len(src.encode()) > args.max_bytes:
            continue
        labs = byte_labels(src, tgt)
        counts.update(labs)
        labelled.append((src, labs))
    vocab = [EMPTY, UNK] + [l for l, _ in counts.most_common(args.vocab) if l != EMPTY][: args.vocab - 2]
    index = {l: i for i, l in enumerate(vocab)}
    covered = sum(c for l, c in counts.items() if l in index) / sum(counts.values())
    (OUT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"{len(labelled)} train rows, {len(counts)} distinct labels, vocab {len(vocab)} covers {100 * covered:.2f}% of bytes")
    print("top labels:", [l for l, _ in counts.most_common(25)])

    random.Random(0).shuffle(labelled)
    with (OUT / "train.jsonl").open("w", encoding="utf-8") as f:
        for src, labs in labelled:
            f.write(json.dumps({"src": src, "labels": [index.get(l, 1) for l in labs]}, ensure_ascii=False) + "\n")
    unk_valid = n_valid = 0
    with (OUT / "valid.jsonl").open("w", encoding="utf-8") as f:
        for src, tgt in valid:
            labs = byte_labels(src, tgt)
            ids = [index.get(l, 1) for l in labs]
            unk_valid += ids.count(1)
            n_valid += len(ids)
            f.write(json.dumps({"src": src, "tgt": tgt, "labels": ids}, ensure_ascii=False) + "\n")
    print(f"valid: {len(valid)} rows, UNK rate {100 * unk_valid / max(1, n_valid):.2f}% of bytes")


if __name__ == "__main__":
    main()
