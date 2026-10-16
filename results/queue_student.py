"""On-device step 2, round 1 (2026-10-15): a ~30M-parameter byte-level student trained from scratch on gold data
(data/student_train.jsonl), selection on data/chat_dev_mix.jsonl. Then phone_bench + evaluate against the teacher."""
import ctypes
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\zyota\Desktop\mn-translit")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
LOG = REPO / "results" / "queue_student.log"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONUNBUFFERED="1")
OUT = sys.argv[1] if len(sys.argv) > 1 else "models/byt5-student-30m"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "d_model=512,d_ff=1536,num_layers=6,num_decoder_layers=2,num_heads=8"

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


name = Path(OUT).name
if not (REPO / OUT / "training_info.json").exists():
    run(["train/train_byt5.py", "--model", "google/byt5-small", "--from-scratch", "--config-overrides", CONFIG,
         "--train", "data/student_train.jsonl", "--valid", "data/chat_dev_mix.jsonl", "--out", OUT,
         "--optim", "adamw_torch_fused", "--lr", "1e-3", "--warmup-ratio", "0.05", "--max-steps", "25000",
         "--eval-steps", "2500", "--patience", "3", "--batch-size", "32", "--grad-accum", "2", "--num-workers", "0",
         "--eval-examples", "444", "--resume"], f"results/train_{name}.log")

if (REPO / OUT / "training_info.json").exists():
    run(["eval/evaluate.py", "--test", "data/chat/valid.jsonl", "--real", "data/real_test.jsonl",
         "--extra-test", "chat_test=data/chat_test.jsonl", "real_dev=data/real_dev.jsonl",
         "--model", "models/byt5-small-chat", "--model-name", "teacher-small-chat",
         "--model", OUT, "--model-name", name, "--num-beams", "1", "--out-dir", f"results/{name}"],
        f"results/eval_{name}.log")
    for ds in ("real_test", "chat_test"):
        run(["eval/bootstrap.py", f"results/{name}/pred_{ds}.csv"], f"results/{name}/bootstrap_{ds}.md")
    run(["tools/phone_bench.py", "--model", OUT, "--threads", "4", "--out", f"results/phone/bench_{name}.md"],
        f"results/phone_bench_{name}.log")
log("queue finished")
