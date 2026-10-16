"""Encoder-only transliteration tagger: a pretrained (pruned) ByT5 encoder + a classification head that emits,
for every input byte, the output string it produces (tools/tagger_data.py). No decoder loop: one forward pass
per word, which is what a keyboard needs.

  python train/train_tagger.py --encoder models/byt5-pruned-4e2d-1024-ft2 --out models/tagger-4e
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import corpus_metrics  # noqa: E402

PAD_LABEL = -100


def read(path):
    with open(ROOT / path, encoding="utf-8-sig") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def encode(src: str):
    return [b + 3 for b in src.encode("utf-8")] + [1]  # ByT5: byte + 3, eos = 1


def decode(src: str, pred_ids, vocab):
    """Join the predicted labels; UNK copies the source character."""
    out, pos = [], 0
    for ch in src:
        n = len(ch.encode("utf-8"))
        lab = vocab[pred_ids[pos]] if pos < len(pred_ids) else "<unk>"
        out.append(ch if lab == "<unk>" else lab)
        pos += n
    return "".join(out)


class Tagger:
    def __init__(self, encoder_dir, n_labels, device, dropout=0.1):
        import torch
        from transformers import T5EncoderModel
        self.torch = torch
        self.encoder = T5EncoderModel.from_pretrained(encoder_dir).to(device)
        d = self.encoder.config.d_model
        self.head = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(d, n_labels)).to(device)
        self.device = device

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.head.parameters())

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(h)

    def train(self):
        self.encoder.train()
        self.head.train()

    def eval(self):
        self.encoder.eval()
        self.head.eval()

    def save(self, out, vocab):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out)
        self.torch.save(self.head[1].state_dict(), out / "head.pt")
        (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")


def batches(rows, batch_size, shuffle, rng):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
        # length-grouped: sort within chunks of 50 batches to limit padding
        chunk = batch_size * 50
        order = sum((sorted(order[i:i + chunk], key=lambda k: len(rows[k]["src"].encode()))
                     for i in range(0, len(order), chunk)), [])
    for s in range(0, len(order), batch_size):
        yield [rows[k] for k in order[s:s + batch_size]]


def collate(batch, torch, device):
    ids = [encode(r["src"]) for r in batch]
    labs = [list(r["labels"]) + [PAD_LABEL] for r in batch]  # eos position is ignored
    n = max(map(len, ids))
    x = torch.zeros((len(ids), n), dtype=torch.long)
    m = torch.zeros((len(ids), n), dtype=torch.long)
    y = torch.full((len(ids), n), PAD_LABEL, dtype=torch.long)
    for i, (a, b) in enumerate(zip(ids, labs)):
        x[i, :len(a)] = torch.tensor(a)
        m[i, :len(a)] = 1
        y[i, :len(b)] = torch.tensor(b)
    return x.to(device), m.to(device), y.to(device)


def evaluate(model, rows, vocab, torch, device, batch_size=64):
    model.eval()
    hyps = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for batch in batches(rows, batch_size, False, None):
            x, m, _ = collate(batch, torch, device)
            pred = model.forward(x, m).argmax(-1).tolist()
            for r, p in zip(batch, pred):
                hyps.append(decode(r["src"], p, vocab))
    model.train()
    return corpus_metrics([r["tgt"] for r in rows], hyps), hyps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="models/byt5-pruned-4e2d-1024-ft2")
    ap.add_argument("--train", default="data/tagger/train.jsonl")
    ap.add_argument("--valid", default="data/tagger/valid.jsonl")
    ap.add_argument("--vocab", default="data/tagger/vocab.json")
    ap.add_argument("--out", default="models/tagger-4e")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    import torch
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = json.loads((ROOT / args.vocab).read_text(encoding="utf-8"))
    train_rows, valid_rows = read(args.train), read(args.valid)
    model = Tagger(args.encoder, len(vocab), device)
    print(f"encoder {sum(p.numel() for p in model.encoder.parameters()) / 1e6:.1f}M + head "
          f"{sum(p.numel() for p in model.head.parameters()) / 1e6:.1f}M params; labels {len(vocab)}; "
          f"train {len(train_rows)} valid {len(valid_rows)} on {device}", flush=True)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    step, best, bad, t0 = 0, float("inf"), 0, time.time()
    log = []
    model.train()
    done = False
    while not done:
        for batch in batches(train_rows, args.batch_size, True, rng):
            x, m, y = collate(batch, torch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model.forward(x, m)
                loss = loss_fn(logits.float().view(-1, logits.shape[-1]), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics, hyps = evaluate(model, valid_rows, vocab, torch, device)
                cer = metrics["cer"]
                print(f"== step {step}: valid CER {100 * cer:.2f} EM {100 * metrics['exact']:.1f} | e.g. "
                      f"{valid_rows[0]['src']} -> {hyps[0]}", flush=True)
                log.append({"step": step, "cer": cer, "exact": metrics["exact"]})
                if cer < best:
                    best, bad = cer, 0
                    model.save(args.out, vocab)
                    (ROOT / args.out / "training_info.json").write_text(json.dumps(
                        {"encoder": args.encoder, "args": vars(args), "best_step": step, "best_cer": cer, "log": log},
                        indent=2), encoding="utf-8")
                else:
                    bad += 1
                    if bad >= args.patience:
                        print("early stop", flush=True)
                        done = True
                        break
            if step >= args.steps:
                done = True
                break
    print(f"best valid CER {100 * best:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
