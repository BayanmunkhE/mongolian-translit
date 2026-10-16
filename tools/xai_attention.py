"""Explainability: which input letters the model looked at for each output letter (decoder cross-attention),
plus the error categories on the real test (what kind of mistakes remain).

  python tools/xai_attention.py --model models/byt5-small-chat2 --text "za bi margaash ugluu irne"
  python tools/xai_attention.py --errors results/chat2/pred_real_test.csv --system small-chat2

Writes results/xai/attention_<n>.png (character-level heat map, cross-attention averaged over decoder layers and
heads; byte weights are summed per character so Cyrillic 2-byte letters show as one cell) and
results/xai/error_categories.md.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "xai"


def char_spans(text: str):
    """[(char, first_byte_index, n_bytes)] for the UTF-8 bytes of `text`."""
    spans, pos = [], 0
    for ch in text:
        n = len(ch.encode("utf-8"))
        spans.append((ch, pos, n))
        pos += n
    return spans


def attention_map(model_dir: str, text: str, n: int):
    import numpy as np
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, attn_implementation="eager").eval()
    enc = tok([text], return_tensors="pt")
    with torch.no_grad():
        gen = model.generate(**enc, num_beams=1, max_new_tokens=128, output_attentions=True,
                             return_dict_in_generate=True)
    out_ids = gen.sequences[0, 1:]  # drop decoder start
    out_text = tok.decode(out_ids, skip_special_tokens=True)
    # cross_attentions: tuple over generated steps; each a tuple over layers of (batch, heads, 1, src_len)
    steps = []
    for step in gen.cross_attentions:
        layers = torch.stack([a[0, :, -1, :] for a in step])  # (layers, heads, src)
        steps.append(layers.mean(dim=(0, 1)).numpy())
    att = np.stack(steps)  # (tgt_tokens, src_tokens)
    src_bytes = text.encode("utf-8")
    att = att[:, :len(src_bytes)]  # drop the eos column
    # sum byte-level weights into characters on both axes
    src_spans = char_spans(text)
    tgt_spans = char_spans(out_text)
    src_mat = np.zeros((att.shape[0], len(src_spans)))
    for j, (_, b, k) in enumerate(src_spans):
        src_mat[:, j] = att[:, b:b + k].sum(axis=1)
    mat = np.zeros((len(tgt_spans), len(src_spans)))
    for i, (_, b, k) in enumerate(tgt_spans):
        rows = src_mat[b:min(b + k, src_mat.shape[0])]
        if len(rows):
            mat[i] = rows.mean(axis=0)
    mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-9)

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(src_spans)), max(4, 0.45 * len(tgt_spans))))
    ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(src_spans)))
    ax.set_xticklabels([c for c, _, _ in src_spans], family="monospace")
    ax.set_yticks(range(len(tgt_spans)))
    ax.set_yticklabels([c for c, _, _ in tgt_spans], family="monospace")
    ax.set_xlabel("input (Latin)")
    ax.set_ylabel("output (Cyrillic)")
    ax.set_title(f"cross-attention: {text} → {out_text}", fontsize=10)
    fig.tight_layout()
    path = OUT / f"attention_{n}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    # top-3 source letters per output letter, as text
    lines = [f"{text} → {out_text}", ""]
    for i, (c, _, _) in enumerate(tgt_spans):
        top = np.argsort(-mat[i])[:3]
        lines.append(f"  {c!r}: " + ", ".join(f"{src_spans[j][0]!r} {mat[i, j]:.2f}" for j in top))
    (OUT / f"attention_{n}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{text} → {out_text}  ({path})")


CATEGORIES = [
    ("ө/о", re.compile(r"[өо]"), re.compile(r"[өо]")),
    ("ү/у", re.compile(r"[үу]"), re.compile(r"[үу]")),
    ("й/и/ь/ъ", re.compile(r"[йиьъ]"), re.compile(r"[йиьъ]")),
]


def error_categories(pred_csv: str, system: str):
    """Classify each wrong prediction by what differs from the reference (several categories can apply)."""
    from eval.metrics import align, norm
    rows = list(csv.DictReader(open(ROOT / pred_csv, encoding="utf-8-sig", newline="")))
    cats, n_wrong = Counter(), 0
    examples = {}
    for r in rows:
        ref, hyp = r["tgt"], r[f"{system}_pred"]
        if norm(ref) == norm(hyp):
            continue
        n_wrong += 1
        found = set()
        if norm(ref, True) == norm(hyp, True):
            found.add("том/жижиг үсэг")
        if re.sub(r"[^\w\s]", "", norm(ref)) == re.sub(r"[^\w\s]", "", norm(hyp)) and "том/жижиг үсэг" not in found:
            found.add("цэг таслал")
        rw, hw = norm(ref).split(), norm(hyp).split()
        if len(rw) != len(hw):
            found.add("үгийн тоо (товчлол/задлах)")
        for a, b in zip(rw, hw):
            if a == b:
                continue
            if re.search(r"[A-Za-z]", a) != re.search(r"[A-Za-z]", b):
                found.add("англи үг / латин хэвээр")
            elif a[0].isupper() and len(a) > 3 and a.lower() != b.lower():
                found.add("хүний нэр, нэршил")
            else:
                pairs = [(x, y) for x, y in align(a, b) if x is not None and y is not None and a[x] != b[y]]
                diff = {(a[x], b[y]) for x, y in pairs}
                tagged = False
                for name, pa, pb in CATEGORIES:
                    if any(pa.match(x) and pb.match(y) for x, y in diff):
                        found.add(name)
                        tagged = True
                if not tagged:
                    if len(a) != len(b):
                        found.add("үсэг дутуу/илүү (эгшиг орхилт, давталт)")
                    else:
                        found.add("бусад үсэг солигдсон")
        if not found:
            found.add("бусад")
        for c in found:
            cats[c] += 1
            examples.setdefault(c, []).append(f"`{r['src']}` → `{hyp}` (зөв: `{ref}`)")
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"# Error categories: {system} on {pred_csv}", "",
             f"{n_wrong} of {len(rows)} messages have at least one error. A message can fall in several categories.", "",
             "| Category | Messages | Share of wrong |", "|---|---:|---:|"]
    for c, k in cats.most_common():
        lines.append(f"| {c} | {k} | {100 * k / n_wrong:.0f}% |")
    lines.append("")
    for c, _ in cats.most_common():
        lines += [f"## {c}", ""] + [f"- {e}" for e in examples[c][:3]] + [""]
    (OUT / "error_categories.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4 + len(cats) + 2]))
    (OUT / "error_categories.json").write_text(json.dumps(dict(cats), ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/byt5-small-chat2")
    ap.add_argument("--text", action="append", default=[], help="input(s) to visualise")
    ap.add_argument("--errors", help="pred_*.csv to categorise")
    ap.add_argument("--system", default="small-chat2")
    args = ap.parse_args(argv)
    for n, text in enumerate(args.text, 1):
        attention_map(args.model, text, n)
    if args.errors:
        error_categories(args.errors, args.system)


if __name__ == "__main__":
    main()
