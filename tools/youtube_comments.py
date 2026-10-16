#!/usr/bin/env python3
"""Pick Latin-script Mongolian YouTube comments as real-test candidates.

Download comments first (no video) with yt-dlp:
    yt-dlp --skip-download --write-comments \
        --extractor-args "youtube:max_comments=600,all,all,all;comment_sort=top" \
        -o "data/raw/yt/%(id)s" <video URL> ...

then:
    python tools/youtube_comments.py --in-dir data/raw/yt --out data/raw/youtube_candidates.jsonl
    python annotate/db.py load data/raw/youtube_candidates.jsonl

Only the comment text is kept (no author names; @mentions are removed).
Older videos (2010-2018) have far more Latin-script comments.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Frequent informal Latin spellings of common Mongolian words; >= 2 hits marks a comment as Mongolian.
MARKERS = set("""
bn bna baina bainaa bnu bnuu bainuu uu vv yu ym yum yumaa yumdaa gej gesen gene genee ene ter bi chi ta bid
mash ih sain saihan saikhan duu duug duund duunuud hairtai hairlaj hairla shuu shu dee daa doo l bol bish
minii mini chini chamd cham nadad nad odoo ungu ungiin hun hvn humuus hvmvvs zaluu zurh zvrh setgel bas
tegeed teged yag uneheer unheer unen goy goe gaihaltai uilmaar uilaad hezee yaj yamar sanagdaj sanaj sanagdaad
sonsood sonsoh sonsdog sonsoj oroh irne ireh yavna baisan baigaa bga bgaa bdag baidag bolno bolood bolson
huuhed huuhdvvd ohin zaluus mongol mongolchuud tanii taniig manai mani olon tom
""".split())
CYR_RE = re.compile(r"[Ѐ-ӿ]")
TOKEN_RE = re.compile(r"[a-zA-Z']+")
MENTION_RE = re.compile(r"@\S+")


def clean_comment(text: str) -> str:
    return re.sub(r"\s+", " ", MENTION_RE.sub(" ", text)).strip()


def is_latin_mongolian(text: str) -> bool:
    if CYR_RE.search(text) or "http" in text:
        return False
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    if not 3 <= len(toks) <= 40 or len(text) > 240:
        return False
    hits = sum(t in MARKERS for t in toks)
    letters = sum(c.isalpha() for c in text)
    return hits >= 2 and hits / len(toks) >= 0.25 and letters / len(text) > 0.6


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(ROOT / "data" / "raw" / "yt"), help="yt-dlp *.info.json files")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "youtube_candidates.jsonl"))
    ap.add_argument("--per-video", type=int, default=30, help="cap per video, so no video dominates")
    args = ap.parse_args(argv)

    per_video, seen = {}, set()
    for p in sorted(Path(args.in_dir).glob("*.info.json")):
        info = json.loads(p.read_text(encoding="utf-8"))
        comments = info.get("comments") or []
        keep = []
        for c in comments:
            text = clean_comment(c.get("text", ""))
            if text.lower() not in seen and is_latin_mongolian(text):
                seen.add(text.lower())
                keep.append(text)
        random.Random(info["id"]).shuffle(keep)
        per_video[info["id"]] = keep[:args.per_video]
        print(f"{info['id']}: {len(comments)} comments, {len(keep)} Latin Mongolian, "
              f"keeping {len(per_video[info['id']])}")

    # Round-robin over videos so consecutive items come from different videos.
    items = []
    while any(per_video.values()):
        for vid, texts in per_video.items():
            if texts:
                items.append({"text": texts.pop(0), "source": f"youtube:{vid}"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} candidates -> {out}")


if __name__ == "__main__":
    main()
