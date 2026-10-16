#!/usr/bin/env python3
"""Fine-tune google/byt5-small (or any HF seq2seq model) for Latin -> Cyrillic Mongolian.

Defaults are sized for a free Colab T4 (~2-3 h):
    python train/train_byt5.py --train data/train.jsonl --valid data/valid.jsonl \
        --out models/byt5-mn --max-steps 6000

Resume after a Colab disconnect (checkpoints in --out):
    python train/train_byt5.py ... --resume

Precision: bf16 on Ampere+ GPUs; fp32 on T4 by default because T5-family models
(T5 v1.1 / mT5 / ByT5) are known to overflow to NaN in fp16. --fp16 forces it.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import edit_distance, norm  # noqa: E402


def read_pairs(path):
    src, tgt = [], []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                src.append(d["src"])
                tgt.append(d["tgt"])
    return {"src": src, "tgt": tgt}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/byt5-small")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--valid", default="data/valid.jsonl")
    ap.add_argument("--out", default="models/byt5-mn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-src-len", type=int, default=256, help="tokens (= bytes for ByT5)")
    ap.add_argument("--max-tgt-len", type=int, default=256)
    ap.add_argument("--max-train-examples", type=int, default=0)
    ap.add_argument("--eval-examples", type=int, default=500, help="valid subset used during training")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs if > 0")
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--patience", type=int, default=4, help="early stopping on valid CER")
    ap.add_argument("--num-beams-eval", type=int, default=1, help="greedy during training for speed")
    ap.add_argument("--fp16", action="store_true", help="force fp16 (risk of NaN with T5 models)")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--config-overrides", default="",
                    help="with --from-scratch: comma-separated key=value config changes (student models)")
    ap.add_argument("--from-scratch", action="store_true",
                    help="random weights with --model's architecture (ablation: what pretraining adds)")
    ap.add_argument("--optim", default="adamw_torch", help="e.g. adamw_torch_fused (a bit faster on GPU)")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="dataloader worker processes; 0 saves RAM (data is pre-tokenized, so no slowdown)")
    ap.add_argument("--resume", action="store_true", help="resume from last checkpoint in --out")
    ap.add_argument("--push-to-hub", default="", help="HF repo id to upload the final model")
    return ap.parse_args(argv)


def pick_precision(args) -> dict:
    if not torch.cuda.is_available():
        return {}
    if args.fp16:
        print("WARNING: fp16 with T5-family models may produce NaN losses.")
        return {"fp16": True}
    major, _ = torch.cuda.get_device_capability()
    if major >= 8 and not args.no_bf16:
        return {"bf16": True}
    return {}  # fp32 (T4 = compute capability 7.5)


def training_args(args, precision) -> Seq2SeqTrainingArguments:
    kwargs = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_tgt_len,
        generation_num_beams=args.num_beams_eval,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        optim=args.optim,
        dataloader_num_workers=args.num_workers,
        group_by_length=True,
        **precision,
    )
    # transformers renamed evaluation_strategy -> eval_strategy
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    # transformers 5 removed warmup_ratio; warmup_steps takes a float in [0, 1) as a ratio instead
    if "warmup_ratio" not in params:
        kwargs["warmup_steps"] = kwargs.pop("warmup_ratio")
    # ... and replaced group_by_length=True with train_sampling_strategy="group_by_length"
    if "group_by_length" not in params and "train_sampling_strategy" in params:
        kwargs["train_sampling_strategy"] = "group_by_length" if kwargs.pop("group_by_length") else "random"
    dropped = [k for k in kwargs if k not in params and not k.endswith("strategy")]
    if dropped:
        print(f"WARNING: this transformers version has no {dropped}; they are ignored")
    kwargs = {k: v for k, v in kwargs.items() if k not in dropped}
    return Seq2SeqTrainingArguments(**kwargs)


def main(argv=None):
    args = parse_args(argv)
    for p in (args.train, args.valid):
        if "real_test" in Path(p).name:
            sys.exit("Refusing to train/validate on the real test set — it is evaluation-only.")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.from_scratch:
        config = AutoConfig.from_pretrained(args.model)
        if args.config_overrides:  # e.g. a small student: d_model=512,d_ff=1536,num_layers=6,num_decoder_layers=2
            for kv in args.config_overrides.split(","):
                k, v = kv.split("=")
                setattr(config, k.strip(), type(getattr(config, k.strip()))(v))
        # transformers 5 reports tie_word_embeddings=True for google/byt5-small and from_config() then ties lm_head
        # to the embedding (std 1.0), so the initial logits are ~N(0, d_model) and the first loss is 200-300
        # instead of ln(384)=5.95; every from-scratch run here stalled in a decoder-only plateau because of it.
        # The released ByT5 is untied: untie, and scale lm_head like the other projections.
        config.tie_word_embeddings = False
        model = AutoModelForSeq2SeqLM.from_config(config)
        assert model.lm_head.weight.data_ptr() != model.shared.weight.data_ptr(), "lm_head is still tied"
        model.lm_head.weight.data.normal_(0.0, config.d_model ** -0.5)
        print(f"from scratch: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Only src/tgt are kept, read per file: synthetic and real files carry different extra fields
    # (e.g. "style"), which one load_dataset("json") call cannot cast to a single schema.
    ds = DatasetDict({split: Dataset.from_dict(read_pairs(path))
                      for split, path in (("train", args.train), ("valid", args.valid))})
    if args.max_train_examples:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(
            range(min(args.max_train_examples, len(ds["train"]))))
    ds["valid"] = ds["valid"].shuffle(seed=args.seed).select(
        range(min(args.eval_examples, len(ds["valid"]))))

    def preprocess(batch):
        enc = tokenizer(batch["src"], max_length=args.max_src_len, truncation=True)
        enc["labels"] = tokenizer(text_target=batch["tgt"], max_length=args.max_tgt_len,
                                  truncation=True)["input_ids"]
        return enc

    tokenized = ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
        hyps = tokenizer.batch_decode(preds, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        errs = sum(edit_distance(norm(r), norm(h)) for r, h in zip(refs, hyps))
        chars = sum(len(norm(r)) for r in refs)
        exact = sum(norm(r) == norm(h) for r, h in zip(refs, hyps))
        return {"cer": errs / max(1, chars), "exact": exact / max(1, len(refs))}

    precision = pick_precision(args)
    print(f"precision: {precision or 'fp32'} | train {len(tokenized['train'])} | "
          f"valid {len(tokenized['valid'])}")

    trainer_kwargs = dict(
        model=model,
        args=training_args(args, precision),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    # transformers >= 4.46 uses processing_class, older versions tokenizer
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    last_ckpt = get_last_checkpoint(args.out) if args.resume and os.path.isdir(args.out) else None
    if last_ckpt:
        print(f"Resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    metrics = trainer.evaluate()
    print(metrics)
    trainer.save_model(args.out)  # best checkpoint (load_best_model_at_end)
    tokenizer.save_pretrained(args.out)
    model.generation_config.max_length = args.max_tgt_len
    model.generation_config.save_pretrained(args.out)
    info = {"base_model": args.model, "args": vars(args), "final_valid": metrics,
            "precision": precision or "fp32", "best_checkpoint": trainer.state.best_model_checkpoint}
    Path(args.out, "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Saved model to {args.out}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
