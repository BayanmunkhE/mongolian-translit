#!/usr/bin/env python3
"""Create or update the public Hugging Face Space that serves the demo.

    python tools/push_space.py                    # -> https://huggingface.co/spaces/<you>/mn-translit
    python tools/push_space.py --name other-name --dry-run

Needs HF_TOKEN (a "Write" token). Uploads only what the app needs: code, the small model and the Wikipedia
index. No annotated data (real_*.jsonl, annotate*.db) and no results are uploaded. The Space is public, so the
model weights become public too.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "models/byt5-small-v4-real"
CODE_DIRS = ["app", "baselines", "translit", "rag"]
EXTRA = {"Dockerfile": "deploy/space.Dockerfile", "README.md": "deploy/hf_space_README.md",
         "requirements-app.txt": "requirements-app.txt",
         "data/rag/passages.jsonl": "data/rag/passages.jsonl",
         "data/rag/multilingual-e5-base.npy": "data/rag/multilingual-e5-base.npy"}
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer_config.json",
               "added_tokens.json", "special_tokens_map.json"]


def files() -> dict[str, Path]:
    out = {}
    for d in CODE_DIRS:
        for f in (ROOT / d).rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                out[f.relative_to(ROOT).as_posix()] = f
    for name in MODEL_FILES:
        if (ROOT / MODEL / name).exists():
            out[f"{MODEL}/{name}"] = ROOT / MODEL / name
    for dest, src in EXTRA.items():
        out[dest] = ROOT / src
    missing = [d for d, p in out.items() if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="mn-translit", help="Space name under your account")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    todo = files()
    size = sum(p.stat().st_size for p in todo.values())
    print(f"{len(todo)} files, {size / 1e9:.2f} GB")
    if args.dry_run:
        for dest in sorted(todo):
            print(" ", dest)
        return

    from huggingface_hub import CommitOperationAdd, HfApi
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Write token from https://huggingface.co/settings/tokens")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src)) for dest, src in todo.items()]
    api.create_commit(repo_id, operations=ops, commit_message="Deploy demo", repo_type="space")
    print(f"Uploaded. Build log and app: https://huggingface.co/spaces/{repo_id}")
    print(f"Direct link once it is running: https://{user.lower()}-{args.name}.hf.space")


if __name__ == "__main__":
    main()
