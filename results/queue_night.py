"""Overnight queue 2026-10-16 (deadline 08:30). Stages, each skipped if it would not finish before the deadline:
1. teacher retrain with the seed2 chat data (models/byt5-small-chat2) + eval vs small-chat
2. rerun the round-3 student evaluation that crashed (chat_test, real_dev)
3. continue round 3 on the refreshed gold mix at lr 1e-4 (models/byt5-pruned-4e2d-1024-ft2) + eval + phone bench
4. copy-pretraining experiment for the narrow 30M student: 4k steps of src->src first (forces the decoder to use
   the encoder), then transliteration (models/byt5-student-30m-v2) + eval + phone bench
5. if 4 learned (dev CER < 30%), the same recipe at d_model 768 (models/byt5-student-55m)
"""
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
DEADLINE = dt.datetime.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime(2026, 10, 16, 8, 30)
LOG = REPO / "results" / "queue_night.log"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONUNBUFFERED="1")
EVAL_SETS = ["--test", "data/chat/valid.jsonl", "--real", "data/real_test.jsonl",
             "--extra-test", "chat_test=data/chat_test.jsonl", "real_dev=data/real_dev.jsonl"]
STUDENT_CFG = "d_model=512,d_ff=1536,num_layers=6,num_decoder_layers=2,num_heads=8"

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


def trained(out):
    return (REPO / out / "training_info.json").exists()


def evaluate(models, out_dir, beams, bench=True):
    args = ["eval/evaluate.py", *EVAL_SETS, "--num-beams", str(beams), "--out-dir", out_dir]
    for name, path in models:
        args += ["--model", path, "--model-name", name]
    run(args, f"results/eval_{Path(out_dir).name}.log")
    for ds in ("real_test", "chat_test", "real_dev"):
        if (REPO / out_dir / f"pred_{ds}.csv").exists():
            run(["eval/bootstrap.py", f"{out_dir}/pred_{ds}.csv"], f"{out_dir}/bootstrap_{ds}.md")
    if bench:
        name = models[-1][0]
        run(["tools/phone_bench.py", "--model", models[-1][1], "--threads", "4",
             "--out", f"results/phone/bench_{name}.md"], f"results/phone_bench_{name}.log")


def dev_cer(out):
    try:
        return float(json.loads((REPO / out / "training_info.json").read_text())["final_valid"]["eval_cer"])
    except Exception:  # noqa: BLE001
        return 9.9


# --- 1. teacher retrain with seed2 data --------------------------------------------------------------------
CHAT2 = "models/byt5-small-chat2"
if not trained(CHAT2) and fits(15):
    run(["train/train_byt5.py", "--model", "models/byt5-small-v4-real", "--train", "data/chat_mix.jsonl",
         "--valid", "data/chat_dev_mix.jsonl", "--out", CHAT2, "--optim", "adamw_torch_fused", "--lr", "1e-4",
         "--max-steps", "1200", "--eval-steps", "100", "--patience", "3", "--batch-size", "8", "--grad-accum", "4",
         "--num-workers", "0", "--eval-examples", "500", "--resume"], "results/train_small-chat2.log")
if trained(CHAT2) and fits(10):
    evaluate([("small-chat", "models/byt5-small-chat"), ("small-chat2", CHAT2)], "results/chat2", 4, bench=False)

# --- 2. round-3 student: the evaluation that crashed ---------------------------------------------------------
R3 = "models/byt5-pruned-4e2d-1024-ft"
if fits(8):
    evaluate([("teacher-small-chat", "models/byt5-small-chat"), ("byt5-pruned-4e2d-1024-ft", R3)],
             "results/byt5-pruned-4e2d-1024-ft", 1, bench=False)

# --- 3. continue round 3 on the refreshed mix ----------------------------------------------------------------
R3B = "models/byt5-pruned-4e2d-1024-ft2"
if not trained(R3B) and fits(80):
    run(["tools/student_data.py"], "results/student_data.log")  # picks up the seed2 chat pairs
    run(["train/train_byt5.py", "--model", R3, "--train", "data/student_train.jsonl",
         "--valid", "data/chat_dev_mix.jsonl", "--out", R3B, "--optim", "adamw_torch_fused", "--lr", "1e-4",
         "--warmup-ratio", "0.02", "--max-steps", "20000", "--eval-steps", "2000", "--patience", "4",
         "--batch-size", "32", "--grad-accum", "2", "--num-workers", "0", "--eval-examples", "456", "--resume"],
        "results/train_byt5-pruned-4e2d-1024-ft2.log")
if trained(R3B) and fits(20):
    evaluate([("teacher-small-chat", "models/byt5-small-chat"), ("byt5-pruned-4e2d-1024-ft2", R3B)],
             "results/byt5-pruned-4e2d-1024-ft2", 1)

# --- 4. copy-pretraining for the narrow student --------------------------------------------------------------
COPY_TRAIN, COPY_VALID = REPO / "data" / "copy_train.jsonl", REPO / "data" / "copy_valid.jsonl"
if not COPY_TRAIN.exists():
    rng = random.Random(0)
    rows = [ln for ln in (REPO / "data" / "student_train.jsonl").read_text(encoding="utf-8").splitlines() if ln]
    rng.shuffle(rows)
    with COPY_TRAIN.open("w", encoding="utf-8") as f:
        for ln in rows[:300000]:
            src = json.loads(ln)["src"]
            f.write(json.dumps({"src": src, "tgt": src}, ensure_ascii=False) + "\n")
    with COPY_VALID.open("w", encoding="utf-8") as f:
        for ln in (REPO / "data" / "chat_dev_mix.jsonl").read_text(encoding="utf-8").splitlines()[:300]:
            if ln:
                src = json.loads(ln)["src"]
                f.write(json.dumps({"src": src, "tgt": src}, ensure_ascii=False) + "\n")
    log("wrote copy_train / copy_valid")


def copy_pretrain_then_translit(cfg, copy_out, final_out, copy_minutes, translit_minutes):
    if not trained(copy_out) and fits(copy_minutes):
        run(["train/train_byt5.py", "--model", "google/byt5-small", "--from-scratch", "--config-overrides", cfg,
             "--train", "data/copy_train.jsonl", "--valid", "data/copy_valid.jsonl", "--out", copy_out,
             "--optim", "adamw_torch_fused", "--lr", "1e-3", "--warmup-ratio", "0.1", "--max-steps", "4000",
             "--eval-steps", "1000", "--patience", "10", "--batch-size", "32", "--grad-accum", "2",
             "--num-workers", "0", "--eval-examples", "300", "--resume"], f"results/train_{Path(copy_out).name}.log")
    if trained(copy_out):
        log(f"{copy_out}: copy-task dev CER {dev_cer(copy_out):.3f}")
    if trained(copy_out) and not trained(final_out) and fits(translit_minutes):
        run(["train/train_byt5.py", "--model", copy_out, "--train", "data/student_train.jsonl",
             "--valid", "data/chat_dev_mix.jsonl", "--out", final_out, "--optim", "adamw_torch_fused",
             "--lr", "1e-3", "--warmup-ratio", "0.05", "--max-steps", "30000", "--eval-steps", "2500",
             "--patience", "4", "--batch-size", "32", "--grad-accum", "2", "--num-workers", "0",
             "--eval-examples", "456", "--resume"], f"results/train_{Path(final_out).name}.log")
    if trained(final_out):
        log(f"{final_out}: dev CER {dev_cer(final_out):.3f}")
        if fits(25):
            name = Path(final_out).name
            evaluate([("teacher-small-chat", "models/byt5-small-chat"), (name, final_out)], f"results/{name}", 1)


copy_pretrain_then_translit(STUDENT_CFG, "models/byt5-student-copy", "models/byt5-student-30m-v2", 12, 75)

# --- 5. wider student, only if the copy trick worked --------------------------------------------------------
if trained("models/byt5-student-30m-v2") and dev_cer("models/byt5-student-30m-v2") < 0.30:
    copy_pretrain_then_translit("d_model=768,d_ff=2048,num_layers=6,num_decoder_layers=2,num_heads=12",
                                "models/byt5-student-copy-768", "models/byt5-student-55m", 15, 95)

log("queue finished; GPU is free")
