"""Build the assets of the static demo in site/ (GitHub Pages; the model runs in the browser via onnxruntime-web).

  python tools/build_site.py                      # -> site/assets/ (git-ignored), then open site/ over http

Writes site/assets/{model.json, <tagger>.onnx, vocab.json, dict.json, bigrams.json, selftest.json}:
  * the keyboard tagger (default models/onnx/lstm-tagger-512x3-ft, int8);
  * the dictionary trimmed to the --keep most frequent Latin words, candidates only;
  * bigrams trimmed to the --follow most frequent next words per previous word;
  * selftest.json: Python outputs (OnnxTaggerTransliterator / convert_mixed / KeyboardSuggester on the same trimmed
    files) that site/selftest.html compares with the browser port, so the JS stays equivalent to the Python code.

Selftest inputs come from data/real_dev.jsonl and hand-written examples, never from the evaluation-only test sets.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXAMPLES = [
    "za bi margaash ugluu irne", "bi meeting-ruu orloo", "sain bnu, chi yu hiij bna?", "margaash uulzy",
    "ok bolloo bayrlalaa", "ene kino үнэхээр goy bna", "Yag odoo ene prototype-g turshij bna",
    "unuudur ih zavgui bailaa, daraa yariya", "tgvl deploy-g 5d hiine", "LG zuragt avmaar bna 😊",
]
KEYBOARD_CASES = [
    ("margaash", [], None), ("uulzy", ["margaash"], "маргааш"), ("bna", ["sain"], "сайн"), ("dr", ["test", "server"], None),
    ("Yag", [], None), ("prototype-g", ["odoo", "ene"], "энэ"), ("turshij", ["ene", "prototype-g"], None), ("meeting", [], None),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tagger", default="models/onnx/lstm-tagger-512x3-ft")
    ap.add_argument("--dict", default="data/keyboard/word_dict.json")
    ap.add_argument("--bigrams", default="data/keyboard/bigrams.json")
    ap.add_argument("--keep", type=int, default=30000, help="dictionary words kept (most frequent first)")
    ap.add_argument("--follow", type=int, default=12, help="next words kept per previous word")
    ap.add_argument("--out", default="site/assets")
    ap.add_argument("--n-dev", type=int, default=60, help="real_dev messages in the self-test")
    args = ap.parse_args(argv)

    from translit.keyboard import KeyboardSuggester
    from translit.mixed import convert_mixed
    from translit.onnx_backend import OnnxTaggerTransliterator

    tagger_dir = ROOT / args.tagger
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model_file = "tagger.int8.onnx"
    shutil.copy(tagger_dir / model_file, out / model_file)
    shutil.copy(tagger_dir / "vocab.json", out / "vocab.json")

    full = json.loads((ROOT / args.dict).read_text(encoding="utf-8"))
    top = sorted(full.items(), key=lambda kv: -kv[1]["count"])[:args.keep]
    slim = {w: [c[0] for c in v["candidates"]] for w, v in top}
    (out / "dict.json").write_text(json.dumps(slim, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    big = json.loads((ROOT / args.bigrams).read_text(encoding="utf-8"))
    big_slim = {p: dict(sorted(f.items(), key=lambda kv: -kv[1])[:args.follow]) for p, f in big.items()}
    (out / "bigrams.json").write_text(json.dumps(big_slim, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    info = {"file": model_file, "name": tagger_dir.name,
            "size_mb": round((out / model_file).stat().st_size / 2**20, 1), "dict_words": len(slim)}
    (out / "model.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    # Python reference outputs on exactly the files the page loads.
    tr = OnnxTaggerTransliterator(str(tagger_dir), threads=2)
    with tempfile.TemporaryDirectory() as tmp:
        dpath = Path(tmp) / "d.json"
        dpath.write_text(json.dumps({w: {"candidates": [[c, 0.0] for c in cs]} for w, cs in slim.items()},
                                    ensure_ascii=False), encoding="utf-8")
        bpath = Path(tmp) / "b.json"
        bpath.write_text(json.dumps(big_slim, ensure_ascii=False), encoding="utf-8")
        kb = KeyboardSuggester(tagger_dir, dict_path=dpath, bigram_path=bpath)
        kb.seq2seq = None
        dev = [json.loads(l)["src"] for l in (ROOT / "data" / "real_dev.jsonl").read_text(encoding="utf-8-sig").splitlines()
               if l.strip()][:args.n_dev]
        msgs = EXAMPLES + dev
        selftest = {
            "messages": [{"src": s, "out": convert_mixed(s.strip(), tr.transliterate)} for s in msgs],
            "keyboard": [{"word": w, "context": c, "prev_cyr": p, "out": kb.suggest(w, c, p)["candidates"]}
                         for w, c, p in KEYBOARD_CASES],
        }
    (out / "selftest.json").write_text(json.dumps(selftest, ensure_ascii=False, indent=1), encoding="utf-8")

    for p in sorted(out.iterdir()):
        print(f"{p.name:22s} {p.stat().st_size / 2**20:6.2f} MB")


if __name__ == "__main__":
    main()
