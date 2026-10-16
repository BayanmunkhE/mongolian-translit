"""Gradio front end for the Hugging Face Space (free CPU tier, no Docker): the same model, candidates and Wikipedia
search as app/main.py, in a Gradio UI.

    python gradio_app.py                         # http://127.0.0.1:7860
    MODEL_DIR=models/byt5-small-chat2 python gradio_app.py
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from baselines import get_baseline  # noqa: E402
from translit.candidates import candidates_mixed, word_alternatives  # noqa: E402
from translit.mixed import convert_mixed  # noqa: E402

DEFAULT_MODELS = ("byt5-small-chat2", "byt5-small-chat", "byt5-small-v4-real", "byt5-mn")
EXAMPLES = ["za bi margaash ugluu irne", "margaash hed tsagt uulzah ve", "unuudur ih zavgui bailaa, daraa yariya",
            "hvvhdvvd surguulidaa yavsan uu", "ene kino үнэхээр goy bna", "ugui yvy, 5d uulzy",
            "chingis haan hezee torson be", "ulaanbaatar hotiin hun am hed ve"]
baseline = get_baseline(os.getenv("BASELINE", "mns5217"))
lock = threading.Lock()


def load_model():
    source = os.getenv("MODEL_DIR") or next((str(ROOT / "models" / m) for m in DEFAULT_MODELS
                                             if (ROOT / "models" / m).exists()), None)
    if not source:
        return None, None
    from translit.model import Seq2SeqTransliterator
    m = Seq2SeqTransliterator(source, device=os.getenv("MODEL_DEVICE", "cpu"), num_beams=4)
    m("sain baina uu")
    return m, f"ByT5-small ({Path(source).name})"


def load_retriever():
    try:
        from rag.retriever import PASSAGES, RETRIEVERS, load_passages
        kind = os.getenv("RAG_RETRIEVER") or ("e5" if (PASSAGES.parent / "multilingual-e5-base.npy").exists() else "bm25")
        if kind == "off" or not PASSAGES.exists():
            return None, []
        passages = load_passages()
        r = RETRIEVERS[kind](passages)
        r.search(["монгол"], k=1)
        return r, passages
    except Exception:  # noqa: BLE001 - search is optional
        return None, []


model, model_name = load_model()
retriever, passages = load_retriever()


def transliterate(text: str):
    text = (text or "").strip()
    if not text:
        return "", "", "", ""
    t0 = time.perf_counter()
    with lock:
        cands = candidates_mixed(text, model, 3) if model else []
    best = cands[0][0] if cands else "(загвар ачаалагдаагүй)"
    alts = word_alternatives([c for c, _ in cands])
    alt_lines = []
    for a in alts:
        word = best.split()[a["index"]]
        alt_lines.append(f"- **{word}** → {' · '.join(a['options'])}")
    cand_md = "\n".join(f"- {c} ({p:.0%})" for c, p in cands[1:]) if len(cands) > 1 else "—"
    info = f"{model_name or 'дүрэм'} · {1000 * (time.perf_counter() - t0):.0f} мс"
    return best, convert_mixed(text, baseline), ("\n".join(alt_lines) or "—") + "\n\nБусад хувилбар:\n" + cand_md, info


def search(text: str):
    text = (text or "").strip()
    if not text or retriever is None:
        return "Хайлтын индекс ачаалагдаагүй.", ""
    with lock:
        converted = convert_mixed(text, model if model else baseline)
        latin, cyr = retriever.search([text, converted], k=3)

    def fmt(res):
        return "\n".join(f"{i + 1}. **{passages[j]['title']}** — {passages[j]['text'][:160]}…" for i, (j, _) in enumerate(res)) or "—"

    return f"**«{text}»**\n\n" + fmt(latin), f"**«{converted}»**\n\n" + fmt(cyr)


with gr.Blocks(title="Латин → Кирилл хөрвүүлэгч") as demo:
    gr.Markdown("# Латин үсгээр бичсэн монгол текстийг кирилл болгох\n"
                "Жишээ: *za bi margaash ugluu irne* → *За би маргааш өглөө ирнэ*. Кирилл үгийг хөндөхгүй, зөвхөн латин хэсгийг хөрвүүлнэ.")
    inp = gr.Textbox(label="Латин, кирилл эсвэл холимог текст", placeholder="sain baina uu, margaash hed tsagt uulzah ve?", lines=2)
    with gr.Row():
        go = gr.Button("Хөрвүүлэх", variant="primary")
        go_search = gr.Button("Хөрвүүлээд Википедиагаас хайх")
    gr.Examples(EXAMPLES, inputs=inp)
    with gr.Row():
        out_model = gr.Textbox(label="Манай загвар", lines=2)
        out_base = gr.Textbox(label="Энгийн дүрэм (MNS 5217:2012 урвуу)", lines=2)
    alts_md = gr.Markdown(label="Ялгаатай үгийн хувилбарууд")
    info = gr.Markdown()
    with gr.Row():
        hits_latin = gr.Markdown(label="Латинаар шууд хайвал")
        hits_cyr = gr.Markdown(label="Кирилл болгоод хайвал")
    gr.Markdown("Судалгааны прототип (AI Engineer курсийн төсөл). Хүний нэр, ховор үг, шинэ хар үгэнд алдаа гарч болно; "
                "чухал текстийг өөрөө шалгаарай. Бичсэн текст хадгалагдахгүй.")
    go.click(transliterate, inp, [out_model, out_base, alts_md, info])
    inp.submit(transliterate, inp, [out_model, out_base, alts_md, info])
    go_search.click(transliterate, inp, [out_model, out_base, alts_md, info]).then(search, inp, [hits_latin, hits_cyr])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
