#!/usr/bin/env python3
"""Pre-fill empty 'todo' items of an annotation database with a model's output, so the annotator
only corrects a draft. Only for training/dev data from NEW videos: real-test references are never
pre-filled with model output (CLAUDE.md), so this refuses data/annotate.db.

    python tools/prefill_drafts.py --db data/annotate_train.db --model models/byt5-mn-v3

Items whose message is already in data/real_test.jsonl are set to 'skip'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from annotate import db  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--model", default=str(ROOT / "models" / "byt5-mn-v3"))
    ap.add_argument("--real-test", default=str(ROOT / "data" / "real_test.jsonl"))
    args = ap.parse_args(argv)
    if Path(args.db).resolve() == (ROOT / "data" / "annotate.db").resolve():
        sys.exit("data/annotate.db holds the real test set; its references are never pre-filled by a model")

    with open(args.real_test, encoding="utf-8-sig") as f:
        test_src = {json.loads(line)["src"].lower() for line in f if line.strip()}
    con = db.connect(args.db)
    items = [dict(r) for r in con.execute("SELECT id, src FROM items WHERE status = 'todo' AND tgt = ''")]
    leaked = [it for it in items if it["src"].lower() in test_src]
    with con:
        for it in leaked:
            con.execute("UPDATE items SET status = 'skip', note = 'in real test' WHERE id = ?", (it["id"],))
    items = [it for it in items if it["src"].lower() not in test_src]

    from translit.model import Seq2SeqTransliterator
    model = Seq2SeqTransliterator(args.model)
    drafts = model.batch([it["src"] for it in items])
    with con:
        for it, tgt in zip(items, drafts):
            con.execute("UPDATE items SET tgt = ? WHERE id = ?", (tgt, it["id"]))
    print(f"pre-filled {len(items)} drafts with {args.model}; skipped {len(leaked)} real-test duplicates")
    print(db.stats(con))


if __name__ == "__main__":
    main()
