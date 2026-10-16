"""v4 experiments 2026-10-11: synthetic v4 (emoji, stretched letters) -> fine-tune on real comments.
Checkpoints and models are chosen on data/real_dev.jsonl (new videos); the real test is only reported."""
import ctypes
import datetime as dt
import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\zyota\Desktop\mn-translit")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
DEADLINE = dt.datetime.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime(2099, 1, 1)
LOG = REPO / "results" / "queue_v4.log"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONUNBUFFERED="1")
COMMON = ["--valid", "data/real_dev.jsonl", "--batch-size", "8", "--grad-accum", "4", "--num-workers", "0",
          "--resume"]

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


def train(name, out, extra, minutes):
    if (REPO / out / "training_info.json").exists():
        log(f"{out} already trained")
        return True
    if not fits(minutes):
        return False
    run(["train/train_byt5.py", *COMMON, *extra, "--out", out], f"results/train_{name}.log")
    return (REPO / out / "training_info.json").exists()


# stage-2 data: real comments oversampled x20 + 10k synthetic v4 pairs, so the model does not forget
mix = REPO / "data" / "v4_real_mix.jsonl"
if not mix.exists():
    def pair(line):  # only src/tgt: the two sources have different extra fields
        d = json.loads(line)
        return json.dumps({"src": d["src"], "tgt": d["tgt"]}, ensure_ascii=False)

    real = [pair(line) for line in (REPO / "data" / "real_train.jsonl").read_text(encoding="utf-8").splitlines()]
    with (REPO / "data" / "v4" / "train.jsonl").open(encoding="utf-8") as f:
        synth = [pair(line) for _, line in zip(range(10000), f)]  # train.jsonl is already shuffled
    rows = real * 20 + synth
    random.Random(42).shuffle(rows)
    mix.write_text("\n".join(rows) + "\n", encoding="utf-8")
    log(f"wrote {mix.name}: {len(real)} real x20 + {len(synth)} synthetic")

done = [("byt5-v3", "models/byt5-mn-v3")]
STAGE1 = ["--train", "data/v4/train.jsonl", "--lr", "3e-4", "--eval-steps", "1000"]
STAGE2 = ["--train", "data/v4_real_mix.jsonl", "--lr", "1e-4", "--max-steps", "600", "--eval-steps", "100",
          "--patience", "3"]

for size, optim, minutes in (("small", "adamw_torch_fused", 65), ("base", "adafactor", 110)):
    s1 = f"models/byt5-{size}-v4"
    if train(f"{size}-v4", s1, [*STAGE1, "--model", f"google/byt5-{size}", "--optim", optim,
                                 "--max-steps", "7000"], minutes):
        done.append((f"{size}-v4", s1))
        s2 = f"models/byt5-{size}-v4-real"
        if train(f"{size}-v4-real", s2, [*STAGE2, "--model", s1, "--optim", optim], 25):
            done.append((f"{size}-v4-real", s2))

if fits(45):
    args = ["eval/evaluate.py", "--test", "data/v4/test.jsonl", "--extra-test", "real_dev=data/real_dev.jsonl",
            "--baselines", "--num-beams", "4", "--out-dir", "results/v4"]
    for name, path in done:
        args += ["--model", path, "--model-name", name]
    run(args, "results/eval_v4.log")
    for ds in ("real_dev", "real_test"):
        run(["eval/bootstrap.py", f"results/v4/pred_{ds}.csv"], f"results/v4/bootstrap_{ds}.md")

log(f"queue finished; GPU is free. trained: {json.dumps(done)}")
