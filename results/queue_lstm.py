"""2026-10-17, after results/queue_tagger.py (deadline 15:00): the LSTM baseline on the same labels.
Waits until queue_tagger.log says "queue finished" so the two never share the GPU.
"""
import ctypes
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\zyota\Desktop\mn-translit")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
DEADLINE = dt.datetime.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime(2026, 10, 17, 15, 0)
LOG = REPO / "results" / "queue_lstm.log"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")

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


other = REPO / "results" / "queue_tagger.log"
log("waiting for queue_tagger")
while dt.datetime.now() < DEADLINE and "queue finished" not in other.read_text(encoding="utf-8"):
    time.sleep(60)
if dt.datetime.now() < DEADLINE:
    run(["train/train_lstm_tagger.py", "--out", "models/lstm-tagger", "--steps", "20000"], "results/train_lstm-tagger.log")
log("queue finished")
