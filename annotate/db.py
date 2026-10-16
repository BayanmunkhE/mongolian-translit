#!/usr/bin/env python3
"""SQLite store behind the annotation UI (Latin message -> Cyrillic reference).

    python annotate/db.py load data/raw/youtube_candidates.jsonl   # or a .txt, one message per line
    python annotate/db.py stats
    python annotate/db.py export                                   # -> data/real_test_todo.csv
    python tools/real_test.py import data/real_test_todo.csv      # the only way into real_test.jsonl

A second database of comments from other videos holds real training/dev data:
    python annotate/db.py --db data/annotate_train.db load data/raw/youtube_train_candidates.jsonl
    python tools/prefill_drafts.py --db data/annotate_train.db --model models/byt5-mn-v3
    python annotate/db.py --db data/annotate_train.db pairs        # -> data/real_train.jsonl, real_dev.jsonl

The database (data/annotate.db, or $ANNOTATE_DB) is a work area, not the test
set: only 'done' items are exported, and tools/real_test.py validates them again.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_pairs import normalize_cyrillic  # noqa: E402
from tools.real_test import CYR_RE, clean, validate  # noqa: E402

DB_PATH = Path(os.getenv("ANNOTATE_DB", ROOT / "data" / "annotate.db"))
STATUSES = ("todo", "done", "skip")
SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    src        TEXT NOT NULL UNIQUE COLLATE NOCASE,  -- Latin message exactly as written
    tgt        TEXT NOT NULL DEFAULT '',             -- Cyrillic reference
    status     TEXT NOT NULL DEFAULT 'todo' CHECK (status IN ('todo', 'done', 'skip')),
    source     TEXT NOT NULL DEFAULT '',             -- e.g. youtube:<video id>, messenger
    note       TEXT NOT NULL DEFAULT '',
    updated_at TEXT
);
"""


def connect(path=None) -> sqlite3.Connection:
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI may open a per-request connection and use it on another worker thread.
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _one(cur) -> dict | None:
    row = cur.fetchone()
    return dict(row) if row else None


def get(con, item_id: int) -> dict | None:
    return _one(con.execute("SELECT * FROM items WHERE id = ?", (item_id,)))


def add(con, src: str, source: str = "", note: str = "") -> int | None:
    """Insert one Latin message. Returns its id, or None if it is already in the database."""
    src = clean(src)
    if not src:
        raise ValueError("empty src")
    if CYR_RE.search(src):
        raise ValueError("src contains Cyrillic — src must be the Latin-script message as written")
    with con:
        cur = con.execute("INSERT OR IGNORE INTO items (src, source, note) VALUES (?, ?, ?)",
                          (src, clean(source), clean(note)))
    return cur.lastrowid if cur.rowcount else None


def load_file(con, path, source: str = "") -> dict:
    """Load candidates from .jsonl ({"text" or "src", optional "source"}) or .txt (one per line)."""
    path = Path(path)
    counts = {"added": 0, "duplicate": 0, "rejected": 0}
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                obj = json.loads(line)
                src, item_source = obj.get("text") or obj.get("src", ""), obj.get("source") or source
            else:
                src, item_source = line, source
            try:
                counts["added" if add(con, src, item_source) else "duplicate"] += 1
            except ValueError:
                counts["rejected"] += 1
    return counts


def next_todo(con, after: int = 0) -> dict | None:
    """First unannotated item after `after`, wrapping around to the start."""
    return (_one(con.execute("SELECT * FROM items WHERE status = 'todo' AND id > ? ORDER BY id LIMIT 1",
                             (after,)))
            or _one(con.execute("SELECT * FROM items WHERE status = 'todo' ORDER BY id LIMIT 1")))


def neighbor(con, item_id: int, step: int) -> dict | None:
    """Previous (step < 0) or next (step > 0) item by id, whatever its status."""
    op, order = (">", "ASC") if step > 0 else ("<", "DESC")
    return _one(con.execute(f"SELECT * FROM items WHERE id {op} ? ORDER BY id {order} LIMIT 1", (item_id,)))


def save(con, item_id: int, tgt: str, status: str, note: str = "") -> tuple[dict, list[str]]:
    """Store a reference; 'done' items must pass tools/real_test.py validation. Returns (item, warnings)."""
    item = get(con, item_id)
    if item is None:
        raise KeyError(item_id)
    if status not in STATUSES:
        raise ValueError(f"bad status {status!r}")
    tgt = normalize_cyrillic(clean(tgt))
    warnings = []
    if status == "done":
        errors, warnings = validate(item["src"], tgt)
        if errors:
            raise ValueError("; ".join(errors))
    with con:
        con.execute("UPDATE items SET tgt = ?, status = ?, note = ?, updated_at = datetime('now') "
                    "WHERE id = ?", (tgt, status, clean(note), item_id))
    return get(con, item_id), warnings


def stats(con) -> dict:
    counts = {r[0]: r[1] for r in con.execute("SELECT status, COUNT(*) FROM items GROUP BY status")}
    out = {s: counts.get(s, 0) for s in STATUSES}
    out["total"] = sum(out.values())
    return out


def write_csv(con, f) -> int:
    """Write 'done' items in the CSV format `tools/real_test.py import` reads. Returns the row count."""
    rows = con.execute("SELECT src, tgt, source, note FROM items WHERE status = 'done' ORDER BY id").fetchall()
    w = csv.writer(f)
    w.writerow(["src", "tgt", "source", "note"])
    w.writerows(tuple(r) for r in rows)
    return len(rows)


def write_pairs(con, out_dir, dev_frac: float = 0.33) -> dict:
    """Write 'done' items as training JSONL (real_train.jsonl / real_dev.jsonl), split by source (video),
    so the dev set never shares a video with the training set. Only for a database of NEW videos, never
    for the real test set."""
    rows = con.execute("SELECT src, tgt, source FROM items WHERE status = 'done' ORDER BY id").fetchall()
    by_source: dict[str, list] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    sources = sorted(by_source, key=lambda s: hashlib.md5(s.encode()).hexdigest())
    dev, train, target = [], [], dev_frac * len(rows)
    for s in sources:
        (dev if len(dev) < target else train).extend(by_source[s])
    out_dir = Path(out_dir)
    for name, part in (("real_train", train), ("real_dev", dev)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"src": r["src"], "tgt": r["tgt"], "source": r["source"]},
                                   ensure_ascii=False) + "\n")
    return {"real_train": len(train), "real_dev": len(dev)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DB_PATH))
    sub = ap.add_subparsers(dest="cmd", required=True)
    lo = sub.add_parser("load", help="add candidate Latin messages")
    lo.add_argument("files", nargs="+")
    lo.add_argument("--source", default="", help="source label for .txt files, e.g. messenger")
    ex = sub.add_parser("export", help="write 'done' items as CSV for tools/real_test.py import")
    ex.add_argument("out", nargs="?", default=str(ROOT / "data" / "real_test_todo.csv"))
    pa = sub.add_parser("pairs", help="write 'done' items as data/real_train.jsonl + real_dev.jsonl (new videos only)")
    pa.add_argument("--out-dir", default=str(ROOT / "data"))
    pa.add_argument("--dev-frac", type=float, default=0.33)
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    if args.cmd == "pairs" and Path(args.db).resolve() == (ROOT / "data" / "annotate.db").resolve():
        sys.exit("data/annotate.db holds the real test set; training pairs must come from another --db")

    con = connect(args.db)
    if args.cmd == "load":
        for p in args.files:
            print(f"{p}: {load_file(con, p, args.source)}")
    elif args.cmd == "export":
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel opens UTF-8 correctly
            n = write_csv(con, f)
        print(f"Wrote {n} done items to {args.out}. Next: python tools/real_test.py import {args.out}")
    elif args.cmd == "pairs":
        print(write_pairs(con, args.out_dir, args.dev_frac))
    print(stats(con))


if __name__ == "__main__":
    main()
