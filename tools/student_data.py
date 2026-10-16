"""Training mix for the on-device student model (round 1: gold labels only, no teacher labelling needed).

  python tools/student_data.py        # -> data/student_train.jsonl

v4 synthetic (1M, gold) + chat pairs x3 (gold) + real_train x20 (gold), shuffled. Validation stays
data/chat_dev_mix.jsonl (real_dev + chat valid). Teacher-labelled text is only worth adding once there is
unlabelled real Latin chat text (user chats); data/raw/youtube_candidates.jsonl is the real_test pool, so it is
never used here.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pairs(path, repeat=1):
    out = []
    with open(ROOT / path, encoding="utf-8-sig") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                out.append(json.dumps({"src": d["src"], "tgt": d["tgt"]}, ensure_ascii=False))
    return out * repeat


rows = pairs("data/v4/train.jsonl") + pairs("data/chat/train.jsonl", 3) + pairs("data/real_train.jsonl", 20)
random.Random(42).shuffle(rows)
out = ROOT / "data" / "student_train.jsonl"
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"{len(rows)} rows -> {out}")
