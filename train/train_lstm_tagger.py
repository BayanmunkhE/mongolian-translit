"""LSTM baseline for the rubric (NLP track: TF-IDF / LSTM / fine-tuned Transformer): a byte-level BiLSTM tagger
trained from scratch on the same byte -> output-string labels as train/train_tagger.py (tools/tagger_data.py).
No pretraining, no attention: it shows what the pretrained ByT5 encoder adds at a similar size.

  python train/train_lstm_tagger.py --train data/tagger_full/train.jsonl --valid data/tagger_full/valid.jsonl \
      --vocab data/tagger_full/vocab.json --out models/lstm-tagger

After training, the best checkpoint is scored on data/real_test.jsonl and data/chat_test.jsonl one message at a
time on the CPU (results/lstm/<out-name>.md).
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
from train.train_tagger import PAD_LABEL, batches, collate, decode, encode, evaluate, read  # noqa: E402


def build(torch, n_labels, d_emb=256, hidden=384, layers=2, dropout=0.2):
    class LstmTagger(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(259, d_emb, padding_idx=0)  # ByT5 byte ids: 0 pad, 1 eos, 3..258 bytes
            self.lstm = torch.nn.LSTM(d_emb, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                                      dropout=dropout)
            self.drop = torch.nn.Dropout(dropout)
            self.head = torch.nn.Linear(2 * hidden, n_labels)

        def forward(self, ids, mask):
            lengths = mask.sum(1).cpu()
            x = self.drop(self.emb(ids))
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            h, _ = self.lstm(packed)
            h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.shape[1])
            return self.head(self.drop(h))

    return LstmTagger()


def score_sets(model, vocab, torch, out_md, name):
    """Real test + chat test, one message at a time on the CPU: accuracy and latency like the ONNX tagger check."""
    model = model.to("cpu").eval()
    torch.set_num_threads(4)
    lines = [f"# LSTM tagger: {name}", "", f"- params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"]
    words = []
    for set_name, path in (("real_test", "data/real_test.jsonl"), ("chat_test", "data/chat_test.jsonl")):
        rows = read(path)
        words += [w for r in rows for w in r["src"].split() if w.isalpha()]
        hyps, t0 = [], time.perf_counter()
        with torch.inference_mode():
            for r in rows:
                ids = torch.tensor([encode(r["src"])])
                pred = model(ids, torch.ones_like(ids)).argmax(-1)[0].tolist()
                hyps.append(decode(r["src"], pred, vocab))
        ms = 1000 * (time.perf_counter() - t0) / len(rows)
        m = corpus_metrics([r["tgt"] for r in rows], hyps)
        lines.append(f"- {set_name}: CER {100 * m['cer']:.2f} WER {100 * m['wer']:.2f} EM {100 * m['exact']:.1f} "
                     f"chrF {100 * m.get('chrf', 0):.1f} | {ms:.1f} ms/message (CPU, 4 threads)")
        (out_md.parent / f"pred_{set_name}_{name}.csv").write_text(
            "\n".join(f"{r['src']}\t{r['tgt']}\t{h}" for r, h in zip(rows, hyps)), encoding="utf-8")
    times = []
    with torch.inference_mode():
        for w in words[:150]:
            ids = torch.tensor([encode(w)])
            t0 = time.perf_counter()
            model(ids, torch.ones_like(ids))
            times.append(1000 * (time.perf_counter() - t0))
    times.sort()
    lines.append(f"- single words: p50 {times[len(times) // 2]:.1f} ms, p95 {times[int(len(times) * .95)]:.1f} ms")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/tagger_full/train.jsonl")
    ap.add_argument("--valid", default="data/tagger_full/valid.jsonl")
    ap.add_argument("--vocab", default="data/tagger_full/vocab.json")
    ap.add_argument("--out", default="models/lstm-tagger")
    ap.add_argument("--d-emb", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-only", action="store_true", help="skip training, score the saved model")
    ap.add_argument("--init", help="continue from this model.pt (same architecture flags) instead of random init")
    args = ap.parse_args(argv)

    import torch
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = json.loads((ROOT / args.vocab).read_text(encoding="utf-8"))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    md = ROOT / "results" / "lstm" / f"{out.name}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    model = build(torch, len(vocab), args.d_emb, args.hidden, args.layers, args.dropout)
    if args.score_only:
        model.load_state_dict(torch.load(out / "model.pt", map_location="cpu"))
        score_sets(model, vocab, torch, md, out.name)
        return
    if args.init:
        model.load_state_dict(torch.load(ROOT / args.init, map_location="cpu"))
    model = model.to(device)
    train_rows, valid_rows = read(args.train), read(args.valid)
    print(f"BiLSTM {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params; labels {len(vocab)}; "
          f"train {len(train_rows)} valid {len(valid_rows)} on {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    step, best, bad, t0, log, done = 0, float("inf"), 0, time.time(), [], False
    model.train()
    while not done:
        for batch in batches(train_rows, args.batch_size, True, rng):
            x, m, y = collate(batch, torch, device)
            logits = model(x, m)  # fp32: cuDNN LSTM has no bf16 path worth the trouble at 9M params
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
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
                    torch.save(model.state_dict(), out / "model.pt")
                    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
                    (out / "training_info.json").write_text(json.dumps(
                        {"args": vars(args), "best_step": step, "best_cer": cer, "log": log}, indent=2), encoding="utf-8")
                else:
                    bad += 1
                    if bad >= args.patience:
                        print("early stop", flush=True)
                        done = True
                        break
            if step >= args.steps:
                done = True
                break
    print(f"best valid CER {100 * best:.2f} -> {args.out}", flush=True)
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    score_sets(model, vocab, torch, md, out.name)


if __name__ == "__main__":
    main()
