"""Stage 3b (2026-10-15): retrain the chat model with the targeted seed2 data (-ъя endings, digit+suffix,
-чих verbs, мин, interjections, split suffixes). Waits for the round-3 student training to finish."""
import ctypes
import datetime as dt
import os
import subprocess
import time
from pathlib import Path

REPO = Path(r"C:\Users\zyota\Desktop\mn-translit")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
LOG = REPO / "results" / "queue_chat2.log"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONUNBUFFERED="1")
OUT = "models/byt5-small-chat2"
WAIT_FOR = REPO / "models" / "byt5-pruned-4e2d-1024-ft" / "training_info.json"

ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # keep Windows awake while running


def log(msg):
    line = f"{dt.datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, out_file):
    log(f"start: {' '.join(args)} > {out_file}")
    with open(REPO / out_file, "w", encoding="utf-8") as f:
        code = subprocess.call([PY, *args], cwd=REPO, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    log(f"done (exit {code}): {out_file}")
    return code


while not WAIT_FOR.exists():
    time.sleep(30)
log("round-3 student training done; starting")

if not (REPO / OUT / "training_info.json").exists():
    run(["train/train_byt5.py", "--model", "models/byt5-small-v4-real", "--train", "data/chat_mix.jsonl",
         "--valid", "data/chat_dev_mix.jsonl", "--out", OUT, "--optim", "adamw_torch_fused", "--lr", "1e-4",
         "--max-steps", "1200", "--eval-steps", "100", "--patience", "3", "--batch-size", "8", "--grad-accum", "4",
         "--num-workers", "0", "--eval-examples", "500", "--resume"], "results/train_small-chat2.log")

if (REPO / OUT / "training_info.json").exists():
    run(["eval/evaluate.py", "--test", "data/chat/valid.jsonl", "--real", "data/real_test.jsonl",
         "--extra-test", "chat_test=data/chat_test.jsonl", "real_dev=data/real_dev.jsonl",
         "--model", "models/byt5-small-chat", "--model-name", "small-chat",
         "--model", OUT, "--model-name", "small-chat2", "--num-beams", "4", "--out-dir", "results/chat2"],
        "results/eval_chat2.log")
    for ds in ("real_test", "chat_test"):
        run(["eval/bootstrap.py", f"results/chat2/pred_{ds}.csv"], f"results/chat2/bootstrap_{ds}.md")
log("queue finished")
