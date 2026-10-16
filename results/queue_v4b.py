"""v4b 2026-10-12: two stage-2 variants after queue_v4 (selection on data/real_dev.jsonl only).
base-v4-real picked checkpoint 100 of 600 at lr 1e-4 -> try lr 3e-5; small-v4-real still improved at 500 -> 1200 steps."""
import ctypes
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\zyota\Desktop\mn-translit")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
DEADLINE = dt.datetime.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime(2099, 1, 1)
LOG = REPO / "results" / "queue_v4b.log"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONUNBUFFERED="1")
STAGE2 = ["--valid", "data/real_dev.jsonl", "--batch-size", "8", "--grad-accum", "4", "--num-workers", "0",
          "--resume", "--train", "data/v4_real_mix.jsonl", "--eval-steps", "100", "--patience", "3"]

ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # keep Windows awake while running


def log(msg):
    line = f"{dt.datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, out_file):
    log(f"start: {' '.join(args)} > {out_file}")
    timeout = (DEADLINE - dt.datetime.now()).total_seconds()
    with open(REPO / out_file, "w", encoding="utf-8") as f:
        p = subprocess.Popen([PY, *args], cwd=REPO, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        try:
            code = p.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            p.kill()
            code = "killed at deadline"
    log(f"done (exit {code}): {out_file}")
    return code


def fits(minutes):
    end = dt.datetime.now() + dt.timedelta(minutes=minutes)
    ok = end <= DEADLINE
    log(f"needs ~{minutes:.0f} min, would end {end:%H:%M} -> {'run' if ok else 'SKIP (deadline)'}")
    return ok


done = [("small-v4-real", "models/byt5-small-v4-real"), ("base-v4-real", "models/byt5-base-v4-real")]
for name, extra, minutes in (
        ("base-v4-real-lr3e-5", ["--model", "models/byt5-base-v4", "--optim", "adafactor", "--lr", "3e-5",
                                 "--max-steps", "600"], 25),
        ("small-v4-real-1200", ["--model", "models/byt5-small-v4", "--optim", "adamw_torch_fused", "--lr", "1e-4",
                                "--max-steps", "1200"], 20)):
    out = f"models/byt5-{name}"
    if not (REPO / out / "training_info.json").exists() and fits(minutes):
        run(["train/train_byt5.py", *STAGE2, *extra, "--out", out], f"results/train_{name}.log")
    if (REPO / out / "training_info.json").exists():
        done.append((name, out))

if fits(50):
    args = ["eval/evaluate.py", "--test", "data/v4/test.jsonl", "--extra-test", "real_dev=data/real_dev.jsonl",
            "--baselines", "--num-beams", "4", "--out-dir", "results/v4b"]
    for name, path in done:
        args += ["--model", path, "--model-name", name]
    (REPO / "results" / "v4b").mkdir(exist_ok=True)
    run(args, "results/eval_v4b.log")
    for ds in ("real_dev", "real_test"):
        run(["eval/bootstrap.py", f"results/v4b/pred_{ds}.csv"], f"results/v4b/bootstrap_{ds}.md")

log(f"queue finished; GPU is free. models: {json.dumps(done)}")
