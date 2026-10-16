"""Structured pruning of a fine-tuned ByT5: keep a subset of encoder/decoder blocks and the most active
feed-forward neurons, so the pretrained attention survives (training a small ByT5 from scratch did not learn
to use the encoder; see PROGRESS.md 2026-10-15).

  python tools/prune_model.py --teacher models/byt5-small-chat --out models/byt5-pruned-4e1d-768 \
      --enc-layers 0,3,7,11 --dec-layers 0 --d-ff 768

Block 0 of each stack must be kept (it owns the relative position bias). Feed-forward neurons are ranked by
their mean |activation| on calibration pairs (hooks on each block's `wo` input), then the top d_ff are copied.
The result needs fine-tuning (train/train_byt5.py --model <out>).
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_pairs(path, n, seed=0):
    rows = [json.loads(ln) for ln in open(ROOT / path, encoding="utf-8-sig") if ln.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def ff_importance(model, tok, pairs, device, batch=32):
    """mean |h| per feed-forward neuron, keyed by (stack, block index); h is the input of `wo`."""
    import torch
    acc, hooks = {}, []
    for stack_name, stack in (("encoder", model.encoder), ("decoder", model.decoder)):
        for i, block in enumerate(stack.block):
            wo = block.layer[-1].DenseReluDense.wo

            def hook(_m, inp, key=(stack_name, i)):
                h = inp[0].detach().float().abs()
                acc[key] = acc.get(key, 0) + h.reshape(-1, h.shape[-1]).sum(0)
            hooks.append(wo.register_forward_pre_hook(hook))
    model.eval()
    with torch.inference_mode():
        for s in range(0, len(pairs), batch):
            chunk = pairs[s:s + batch]
            enc = tok([p["src"] for p in chunk], return_tensors="pt", padding=True, truncation=True, max_length=256)
            lab = tok([p["tgt"] for p in chunk], return_tensors="pt", padding=True, truncation=True, max_length=256)
            labels = lab.input_ids.masked_fill(lab.attention_mask == 0, -100)
            model(**{k: v.to(device) for k, v in enc.items()}, labels=labels.to(device))
    for h in hooks:
        h.remove()
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="models/byt5-small-chat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--enc-layers", default="0,3,7,11")
    ap.add_argument("--dec-layers", default="0")
    ap.add_argument("--d-ff", type=int, default=768)
    ap.add_argument("--calib", default="data/student_train.jsonl")
    ap.add_argument("--calib-examples", type=int, default=2000)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_keep = [int(x) for x in args.enc_layers.split(",")]
    dec_keep = [int(x) for x in args.dec_layers.split(",")]
    assert enc_keep[0] == 0 and dec_keep[0] == 0, "block 0 carries the relative attention bias and must be kept"

    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(args.teacher).to(device)
    print(f"teacher: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M parameters")
    imp = ff_importance(teacher, tok, read_pairs(args.calib, args.calib_examples), device)

    config = AutoConfig.from_pretrained(args.teacher)
    config.num_layers, config.num_decoder_layers, config.d_ff = len(enc_keep), len(dec_keep), args.d_ff
    student = AutoModelForSeq2SeqLM.from_config(config)
    sd = teacher.state_dict()
    new = {}
    for k in ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight",
              "encoder.final_layer_norm.weight", "decoder.final_layer_norm.weight"):
        if k in sd:
            new[k] = sd[k]
    for stack, keep in (("encoder", enc_keep), ("decoder", dec_keep)):
        for new_i, old_i in enumerate(keep):
            idx = torch.topk(imp[(stack, old_i)], args.d_ff).indices.sort().values.cpu()
            prefix_old, prefix_new = f"{stack}.block.{old_i}.", f"{stack}.block.{new_i}."
            for k, v in sd.items():
                if not k.startswith(prefix_old):
                    continue
                nk = prefix_new + k[len(prefix_old):]
                if "relative_attention_bias" in k and new_i != 0:
                    continue  # only block 0 has it
                if k.endswith("DenseReluDense.wi_0.weight") or k.endswith("DenseReluDense.wi_1.weight"):
                    v = v[idx, :]
                elif k.endswith("DenseReluDense.wo.weight"):
                    v = v[:, idx]
                new[nk] = v
    missing, unexpected = student.load_state_dict(new, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    out = ROOT / args.out
    student.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"student: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M parameters -> {out}")
    (out / "pruning.json").write_text(json.dumps({"teacher": args.teacher, "enc_layers": enc_keep,
                                                  "dec_layers": dec_keep, "d_ff": args.d_ff}, indent=2))


if __name__ == "__main__":
    main()
