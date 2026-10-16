"""Copy the on-device model and the dictionary into the Android app's assets (not committed: ~50 MB).

  python tools/pack_keyboard_assets.py --onnx models/onnx/pruned-4e2d-1024-ft2 --dict data/keyboard/word_dict.json

The dictionary is trimmed to {word: {"candidates": [[cyr, prob], ...]}} for the top --keep words.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "keyboard" / "android" / "app" / "src" / "main" / "assets"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="models/onnx/pruned-4e2d-1024-ft2")
    ap.add_argument("--dict", default="data/keyboard/word_dict.json")
    ap.add_argument("--tagger", default="models/onnx/tagger-4e")
    ap.add_argument("--keep", type=int, default=30000)
    args = ap.parse_args(argv)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "meta.json"):
        shutil.copy(ROOT / args.onnx / name, ASSETS / name)
    tagger = ROOT / args.tagger
    if (tagger / "tagger.int8.onnx").exists():  # one-pass tagger: the keyboard's primary path when present
        shutil.copy(tagger / "tagger.int8.onnx", ASSETS / "tagger.int8.onnx")
        shutil.copy(tagger / "vocab.json", ASSETS / "tagger_vocab.json")
    full = json.loads((ROOT / args.dict).read_text(encoding="utf-8"))
    top = sorted(full.items(), key=lambda kv: -kv[1]["count"])[:args.keep]
    slim = {w: {"candidates": v["candidates"]} for w, v in top}
    (ASSETS / "word_dict.json").write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    for p in sorted(ASSETS.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
