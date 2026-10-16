"""Overnight 2026-10-17 (deadline 15:00): the tagger on the full gold mix.
1. labels for all ~1.05M rows (tools/tagger_data.py --max-train 1100000) -> data/tagger_full/
2. train/train_tagger.py 40k steps from the pruned encoder -> models/tagger-4e-full
3. export to ONNX int8 + benchmark; word-level context eval with the tagger
"""
import ctypes
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\zyota\Desktop\mn-translit")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
DEADLINE = dt.datetime.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime(2026, 10, 17, 15, 0)
LOG = REPO / "results" / "queue_tagger.log"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONUNBUFFERED="1")
OUT = "models/tagger-4e-full"

ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def log(msg):
    line = f"{dt.datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, out_file):
    log(f"start: {' '.join(args)} > {out_file}")
    timeout = (DEADLINE - dt.datetime.now()).total_seconds()
    with open(REPO / out_file, "w", encoding="utf-8") as f:
        p = subprocess.Popen([PY, "-X", "utf8", *args], cwd=REPO, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        try:
            code = p.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            p.kill()
            code = "killed at deadline"
    log(f"done (exit {code}): {out_file}")
    return code


data = REPO / "data" / "tagger_full"
if not (data / "train.jsonl").exists():
    run(["tools/tagger_data.py", "--max-train", "1100000", "--vocab", "4000"], "results/tagger_data_full.log")
    # tagger_data writes to data/tagger; move the full set aside so the 400k set stays
    (REPO / "data" / "tagger").rename(data)
if not (REPO / OUT / "training_info.json").exists():
    run(["train/train_tagger.py", "--encoder", "models/byt5-pruned-4e2d-1024-ft2", "--train", "data/tagger_full/train.jsonl",
         "--valid", "data/tagger_full/valid.jsonl", "--vocab", "data/tagger_full/vocab.json", "--out", OUT,
         "--steps", "40000", "--eval-every", "2000", "--patience", "5", "--warmup", "1000"], "results/train_tagger-4e-full.log")
if (REPO / OUT / "training_info.json").exists():
    run(["tools/export_tagger_onnx.py", "--model", OUT, "--out", "models/onnx/tagger-4e-full", "--check"],
        "results/export_tagger-4e-full.log")
    run(["tools/context_eval.py", "--onnx-tagger", "models/onnx/tagger-4e-full"], "results/context_eval_tagger.log")
log("queue finished")
