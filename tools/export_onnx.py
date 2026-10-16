"""Export the on-device student to ONNX for the keyboard (Android: ONNX Runtime Mobile).

  python tools/export_onnx.py --model models/byt5-pruned-4e2d-1024-ft --out models/onnx/pruned-4e2d-1024
  python tools/export_onnx.py ... --check            # compare with PyTorch greedy on test words, time it

Two graphs, no KV cache (the keyboard converts single words of <= ~25 bytes, so re-running the 2-layer decoder
over the growing prefix is cheap and the export stays simple):
  encoder.onnx  : input_ids, attention_mask -> encoder_hidden_states
  decoder.onnx  : decoder_input_ids, encoder_hidden_states, attention_mask -> logits (last position is used)
Both are also saved as dynamic-int8 (MatMul weights) *.int8.onnx with onnxruntime's quantizer.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Encoder:
    def __init__(self, model):
        self.m = model

    def __call__(self, input_ids, attention_mask):
        return self.m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class DecoderStep:
    def __init__(self, model):
        self.m = model

    def __call__(self, decoder_input_ids, encoder_hidden_states, attention_mask):
        out = self.m.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=attention_mask, use_cache=False)
        h = out.last_hidden_state
        if self.m.config.tie_word_embeddings:
            h = h * (self.m.model_dim ** -0.5)
        return self.m.lm_head(h)


def export(args):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    for p in model.parameters():  # the tracer refuses parameters that require grad
        p.requires_grad_(False)
    enc_mod, dec_mod = torch.nn.Module(), torch.nn.Module()
    enc_mod.forward, dec_mod.forward = Encoder(model), DecoderStep(model)
    ids = tok(["sain bnu"], return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(enc_mod, (ids.input_ids, ids.attention_mask), str(out / "encoder.onnx"),
                          input_names=["input_ids", "attention_mask"], output_names=["encoder_hidden_states"],
                          dynamic_axes={"input_ids": {0: "b", 1: "src"}, "attention_mask": {0: "b", 1: "src"},
                                        "encoder_hidden_states": {0: "b", 1: "src"}}, opset_version=17, dynamo=False)
        hidden = enc_mod(ids.input_ids, ids.attention_mask)
        dec_ids = torch.tensor([[model.config.decoder_start_token_id, 100]])
        torch.onnx.export(dec_mod, (dec_ids, hidden, ids.attention_mask), str(out / "decoder.onnx"),
                          input_names=["decoder_input_ids", "encoder_hidden_states", "attention_mask"],
                          output_names=["logits"],
                          dynamic_axes={"decoder_input_ids": {0: "b", 1: "tgt"}, "encoder_hidden_states": {0: "b", 1: "src"},
                                        "attention_mask": {0: "b", 1: "src"}, "logits": {0: "b", 1: "tgt"}},
                          opset_version=17, dynamo=False)
    import onnx
    from onnx import numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for name in ("encoder", "decoder"):
        # the tracer embedded the weights as Constant nodes; the quantizer only quantizes initializers
        path = out / f"{name}.onnx"
        m = onnx.load(str(path))
        keep = []
        for node in m.graph.node:
            if node.op_type == "Constant" and node.attribute and node.attribute[0].name == "value":
                t = node.attribute[0].t
                if len(t.dims) == 2 and t.dims[0] * t.dims[1] >= 4096:
                    t.name = node.output[0]
                    m.graph.initializer.append(t)
                    continue
            keep.append(node)
        del m.graph.node[:]
        m.graph.node.extend(keep)
        onnx.save(m, str(path))
        quantize_dynamic(str(path), str(out / f"{name}.int8.onnx"), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul"])
    tok.save_pretrained(out)
    meta = {"model": args.model, "decoder_start_token_id": model.config.decoder_start_token_id,
            "eos_token_id": model.config.eos_token_id, "pad_token_id": model.config.pad_token_id,
            "vocab_size": model.config.vocab_size, "note": "ByT5 bytes: token = byte + 3; decode greedy, no cache"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    for p in sorted(out.glob("*.onnx")):
        print(f"{p.name}: {p.stat().st_size / 2**20:.1f} MB")


class OnnxTransliterator:
    """Greedy byte-level decoding with the two ONNX graphs (what the Android app will do in Kotlin)."""

    def __init__(self, out_dir, int8=True, threads=4):
        import onnxruntime as ort
        import numpy as np
        self.np = np
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        suffix = ".int8.onnx" if int8 else ".onnx"
        self.enc = ort.InferenceSession(str(out_dir / f"encoder{suffix}"), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(out_dir / f"decoder{suffix}"), so, providers=["CPUExecutionProvider"])
        self.meta = json.loads((out_dir / "meta.json").read_text())

    def __call__(self, text, max_new=64):
        np = self.np
        ids = np.array([[b + 3 for b in text.encode()] + [self.meta["eos_token_id"]]], dtype=np.int64)
        mask = np.ones_like(ids)
        hidden = self.enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        dec = [self.meta["decoder_start_token_id"]]
        for _ in range(max_new):
            logits = self.dec.run(None, {"decoder_input_ids": np.array([dec], dtype=np.int64),
                                         "encoder_hidden_states": hidden, "attention_mask": mask})[0]
            nxt = int(logits[0, -1].argmax())
            if nxt == self.meta["eos_token_id"]:
                break
            dec.append(nxt)
        return bytes(t - 3 for t in dec[1:] if t >= 3).decode("utf-8", errors="replace")


def check(args):
    import torch
    from translit.model import Seq2SeqTransliterator
    torch.set_num_threads(4)
    words = []
    for p in ["data/chat_test.jsonl", "data/real_test.jsonl"]:
        for ln in open(ROOT / p, encoding="utf-8-sig"):
            if ln.strip():
                words += [w for w in json.loads(ln)["src"].split() if w.isalpha()]
    words = words[:120]
    ref = Seq2SeqTransliterator(args.model, device="cpu", num_beams=1)
    t0 = time.perf_counter()
    expect = [ref(w) for w in words]
    t_pt = (time.perf_counter() - t0) / len(words) * 1000
    for int8 in (False, True):
        ox = OnnxTransliterator(ROOT / args.out, int8=int8)
        t0 = time.perf_counter()
        got = [ox(w) for w in words]
        t_ox = (time.perf_counter() - t0) / len(words) * 1000
        agree = sum(g == e for g, e in zip(got, expect))
        print(f"onnx int8={int8}: agrees with PyTorch greedy on {agree}/{len(words)} words; "
              f"{t_ox:.0f} ms/word (PyTorch fp32 {t_pt:.0f} ms/word)")
        for g, e, w in list(zip(got, expect, words))[:3]:
            print(f"   {w} -> onnx '{g}' | torch '{e}'")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/byt5-pruned-4e2d-1024-ft2")
    ap.add_argument("--out", default="models/onnx/pruned-4e2d-1024-ft2")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if not (ROOT / args.out / "decoder.onnx").exists():
        export(args)
    if args.check:
        check(args)


if __name__ == "__main__":
    main()
