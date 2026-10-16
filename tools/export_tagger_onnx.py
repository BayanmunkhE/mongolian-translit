"""Export the encoder-only tagger (train/train_tagger.py) to one ONNX graph and benchmark it per word.

  python tools/export_tagger_onnx.py --model models/tagger-4e --out models/onnx/tagger-4e --check

Graph: input_ids, attention_mask -> label_ids (argmax over the head, per byte). Decoding is a table lookup
(vocab.json) and needs no loop, so a word costs one forward pass. Also writes tagger.int8.onnx (MatMul weights
quantized) and copies vocab.json.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_tagger import decode, encode  # noqa: E402


class Wrapped:
    def __init__(self, encoder, head):
        self.encoder, self.head = encoder, head

    def __call__(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(h).argmax(-1)


def export(args):
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import T5EncoderModel
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    vocab = json.loads((ROOT / args.model / "vocab.json").read_text(encoding="utf-8"))
    enc = T5EncoderModel.from_pretrained(ROOT / args.model).eval()
    head = torch.nn.Linear(enc.config.d_model, len(vocab))
    head.load_state_dict(torch.load(ROOT / args.model / "head.pt", map_location="cpu"))
    head.eval()
    for p in list(enc.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    mod = torch.nn.Module()
    mod.forward = Wrapped(enc, head)
    ids = torch.tensor([encode("sain bnu")])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        torch.onnx.export(mod, (ids, mask), str(out / "tagger.onnx"), input_names=["input_ids", "attention_mask"],
                          output_names=["label_ids"], dynamic_axes={"input_ids": {0: "b", 1: "src"},
                                                                     "attention_mask": {0: "b", 1: "src"},
                                                                     "label_ids": {0: "b", 1: "src"}},
                          opset_version=17, dynamo=False)
    m = onnx.load(str(out / "tagger.onnx"))
    keep = []
    for node in m.graph.node:  # weights come out as Constant nodes; the quantizer wants initializers
        if node.op_type == "Constant" and node.attribute and node.attribute[0].name == "value":
            t = node.attribute[0].t
            if len(t.dims) == 2 and t.dims[0] * t.dims[1] >= 4096:
                t.name = node.output[0]
                m.graph.initializer.append(t)
                continue
        keep.append(node)
    del m.graph.node[:]
    m.graph.node.extend(keep)
    onnx.save(m, str(out / "tagger.onnx"))
    quantize_dynamic(str(out / "tagger.onnx"), str(out / "tagger.int8.onnx"), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"])
    shutil.copy(ROOT / args.model / "vocab.json", out / "vocab.json")
    for p in sorted(out.glob("*.onnx")):
        print(f"{p.name}: {p.stat().st_size / 2**20:.1f} MB")


class OnnxTagger:
    def __init__(self, out_dir, int8=True, threads=4):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        name = "tagger.int8.onnx" if int8 else "tagger.onnx"
        self.sess = ort.InferenceSession(str(out_dir / name), so, providers=["CPUExecutionProvider"])
        self.vocab = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))

    def __call__(self, text):
        ids = self.np.array([encode(text)], dtype=self.np.int64)
        pred = self.sess.run(None, {"input_ids": ids, "attention_mask": self.np.ones_like(ids)})[0][0]
        return decode(text, pred.tolist(), self.vocab)


def check(args):
    from eval.metrics import corpus_metrics
    sets = {}
    for name, path in (("chat_test", "data/chat_test.jsonl"), ("real_test", "data/real_test.jsonl")):
        rows = [json.loads(ln) for ln in open(ROOT / path, encoding="utf-8-sig") if ln.strip()]
        sets[name] = rows
    words = [w for r in sets["chat_test"] + sets["real_test"] for w in r["src"].split() if w.isalpha()][:150]
    lines = [f"# Tagger ONNX benchmark: {args.model}", ""]
    for int8 in (False, True):
        tg = OnnxTagger(ROOT / args.out, int8=int8)
        for name, rows in sets.items():
            t0 = time.perf_counter()
            hyps = [tg(r["src"]) for r in rows]
            ms = 1000 * (time.perf_counter() - t0) / len(rows)
            m = corpus_metrics([r["tgt"] for r in rows], hyps)
            lines.append(f"- int8={int8} {name}: CER {100 * m['cer']:.2f} EM {100 * m['exact']:.1f} | {ms:.0f} ms/message")
        times = []
        for w in words:
            t0 = time.perf_counter()
            tg(w)
            times.append(1000 * (time.perf_counter() - t0))
        times.sort()
        lines.append(f"- int8={int8} single words: p50 {times[len(times) // 2]:.1f} ms, p95 {times[int(len(times) * .95)]:.1f} ms")
    print("\n".join(lines))
    (ROOT / "results" / "phone" / f"bench_{Path(args.model).name}_onnx.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/tagger-4e")
    ap.add_argument("--out", default="models/onnx/tagger-4e")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if not (ROOT / args.out / "tagger.onnx").exists():
        export(args)
    if args.check:
        check(args)


if __name__ == "__main__":
    main()
