"""Copy-pretraining for the narrow from-scratch student, with the init fix (2026-10-16 06:30):
1. finish models/byt5-student-copy2 (src -> src, 4000 steps); 2. if its copy CER < 20%, train transliteration from it
(models/byt5-student-30m-v5, 25k steps); 3. eval vs the teacher + phone bench. Deadline-guarded like queue_night."""
import ctypes
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\zyota\Desktop\mn-translit")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
DEADLINE = dt.datetime.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime(2026, 10, 16, 8, 20)
LOG = REPO / "results" / "queue_copy.log"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONUNBUFFERED="1")
CFG = "d_model=512,d_ff=1536,num_layers=6,num_decoder_layers=2,num_heads=8"
COPY, OUT = "models/byt5-student-copy2", "models/byt5-student-30m-v5"

ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def log(msg):
    line = f"{dt.datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, out_file):
    log(f"start: {' '.join(args)} > {out_file}")
    timeout = (DEADLINE - dt.datetime.now()).total_seconds()
    with open(REPO / out_file, "a", encoding="utf-8") as f:
        p = subprocess.Popen([PY, *args], cwd=REPO, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        try:
            code = p.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            p.kill()
            code = "killed at deadline"
    log(f"done (exit {code}): {out_file}")


def fits(minutes):
    ok = dt.datetime.now() + dt.timedelta(minutes=minutes) <= DEADLINE
    log(f"needs ~{minutes} min -> {'run' if ok else 'SKIP (deadline)'}")
    return ok


def trained(out):
    return (REPO / out / "training_info.json").exists()


def dev_cer(out):
    return float(json.loads((REPO / out / "training_info.json").read_text())["final_valid"]["eval_cer"])


if not trained(COPY) and fits(10):
    run(["train/train_byt5.py", "--model", "google/byt5-small", "--from-scratch", "--config-overrides", CFG,
         "--train", "data/copy_train.jsonl", "--valid", "data/copy_valid.jsonl", "--out", COPY,
         "--optim", "adamw_torch_fused", "--lr", "1e-3", "--warmup-ratio", "0.1", "--max-steps", "4000",
         "--eval-steps", "500", "--patience", "10", "--batch-size", "32", "--grad-accum", "2", "--num-workers", "0",
         "--eval-examples", "300", "--resume"], "results/train_byt5-student-copy2.log")
if trained(COPY):
    log(f"copy task dev CER: {dev_cer(COPY):.3f}")
    if dev_cer(COPY) < 0.20 and not trained(OUT) and fits(50):
        run(["train/train_byt5.py", "--model", COPY, "--train", "data/student_train.jsonl",
             "--valid", "data/chat_dev_mix.jsonl", "--out", OUT, "--optim", "adamw_torch_fused", "--lr", "1e-3",
             "--warmup-ratio", "0.05", "--max-steps", "25000", "--eval-steps", "2500", "--patience", "4",
             "--batch-size", "32", "--grad-accum", "2", "--num-workers", "0", "--eval-examples", "444", "--resume"],
            "results/train_byt5-student-30m-v5.log")
if trained(OUT):
    log(f"translit dev CER: {dev_cer(OUT):.3f}")
    if fits(15):
        name = Path(OUT).name
        run(["eval/evaluate.py", "--test", "data/chat/valid.jsonl", "--real", "data/real_test.jsonl",
             "--extra-test", "chat_test=data/chat_test.jsonl", "real_dev=data/real_dev.jsonl",
             "--model", "models/byt5-small-chat", "--model-name", "teacher-small-chat", "--model", OUT,
             "--model-name", name, "--num-beams", "1", "--out-dir", f"results/{name}"], f"results/eval_{name}.log")
        for ds in ("real_test", "chat_test"):
            run(["eval/bootstrap.py", f"results/{name}/pred_{ds}.csv"], f"results/{name}/bootstrap_{ds}.md")
        run(["tools/phone_bench.py", "--model", OUT, "--threads", "4", "--out", f"results/phone/bench_{name}.md"],
            f"results/phone_bench_{name}.log")
log("queue finished")
