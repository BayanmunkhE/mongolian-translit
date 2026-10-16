"""Overnight experiments 2026-10-10 -> 11. All GPU work must finish before DEADLINE (user wakes at 11:00)."""
import ctypes
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

REPO = Path(r"C:\Users\zyota\Desktop\mn-translit")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
DEADLINE = dt.datetime(2026, 10, 11, 10, 45)
LOG = REPO / "results" / "queue_overnight.log"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
TRAIN = ["train/train_byt5.py", "--train", "data/v3/train.jsonl", "--valid", "data/v3/valid.jsonl",
         "--batch-size", "8", "--grad-accum", "4", "--eval-steps", "1000", "--num-workers", "0",
         "--lr", "3e-4", "--resume"]

# keep Windows awake while this script runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); no settings changed
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


def train_and_eval(name, model_dir, extra, minutes):
    if (REPO / model_dir / "training_info.json").exists():
        log(f"{model_dir} already trained, skipping training")
    elif not fits(minutes):
        return None
    else:
        t0 = dt.datetime.now()
        run([*TRAIN, *extra, "--out", model_dir], f"results/train_{name}.log")
        if not (REPO / model_dir / "training_info.json").exists():
            log(f"{name}: no final model, skipping eval")
            return None
        log(f"{name}: training took {(dt.datetime.now() - t0).total_seconds() / 60:.0f} min")
    if fits(15):
        run(["eval/evaluate.py", "--test", "data/v3/valid.jsonl", "--baselines", "--model", model_dir,
             "--model-name", name, "--num-beams", "4", "--out-dir", f"results/{name}"], f"results/eval_{name}.log")
    return model_dir


done = []
log("overnight queue start")

# 1. method 6: ByT5-base, same budget as small v3 (Adafactor: AdamW state for 580M params spills out of 12 GB)
t0 = dt.datetime.now()
if train_and_eval("byt5-base", "models/byt5-base-v3",
                  ["--model", "google/byt5-base", "--optim", "adafactor", "--max-steps", "7000"], 150):
    done.append(("byt5-base", "models/byt5-base-v3"))
base_min_per_kstep = None
if done:
    # seconds per step measured from the 7k run (includes the 7 in-training evals)
    base_min_per_kstep = (dt.datetime.now() - t0).total_seconds() / 60 / 7 * 0.97
    log(f"byt5-base: ~{base_min_per_kstep:.1f} min per 1000 steps incl. evals")

# 2. v3 again with the warmup / length-grouping fix (v1-v3 trained without them)
if train_and_eval("byt5-v3-warmup", "models/byt5-mn-v3-warmup",
                  ["--model", "google/byt5-small", "--optim", "adamw_torch_fused", "--max-steps", "7000"], 75):
    done.append(("byt5-v3-warmup", "models/byt5-mn-v3-warmup"))

# 3. small with Adafactor, to separate the optimizer change from the model size in (1)
if train_and_eval("byt5-v3-adafactor", "models/byt5-mn-v3-adafactor",
                  ["--model", "google/byt5-small", "--optim", "adafactor", "--max-steps", "7000"], 75):
    done.append(("byt5-v3-adafactor", "models/byt5-mn-v3-adafactor"))

# 4. method 5 on the base model: longer training, steps sized to the time left
if base_min_per_kstep:
    left = (DEADLINE - dt.datetime.now()).total_seconds() / 60 - 60  # final eval + margin
    ksteps = min(20, int(left / base_min_per_kstep))
    log(f"base long run: {left:.0f} min available -> {ksteps}k steps")
    if ksteps >= 10 and train_and_eval(
            "byt5-base-long", "models/byt5-base-v3-long",
            ["--model", "google/byt5-base", "--optim", "adafactor", "--max-steps", str(ksteps * 1000)],
            ksteps * base_min_per_kstep + 5):
        done.append((f"byt5-base-{ksteps}k", "models/byt5-base-v3-long"))

# 5. everything on the synthetic test + real test in one file, so the bootstrap is paired
if done and fits(40):
    args = ["eval/evaluate.py", "--test", "data/v3/test.jsonl", "--baselines", "--model", "models/byt5-mn-v3",
            "--model-name", "byt5-v3", "--num-beams", "4", "--out-dir", "results/overnight"]
    for name, path in done:
        args += ["--model", path, "--model-name", name]
    run(args, "results/eval_overnight.log")
    run(["eval/bootstrap.py", "results/overnight/pred_real_test.csv", "--refs", "data/real_test.jsonl"],
        "results/overnight/bootstrap_real_test.md")

log(f"queue finished; GPU is free. trained: {json.dumps(done)}")
