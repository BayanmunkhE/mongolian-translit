# CLAUDE.md

Capstone project (AI Engineer course, NLP track, 14 days): fine-tune `google/byt5-small` to
convert informal Latin-script Mongolian into Cyrillic (`za bi margaash ugluu irne` →
`За би маргааш өглөө ирнэ`), keep English words unchanged, beat rule baselines, and serve
it in a web demo. The user works on Windows and a MacBook; training runs on free Colab T4.
See README.md for the full pipeline and PROGRESS.md for the plan and status.

## Commands

```bash
pytest -q                                              # fast tests, no torch needed
python generate_pairs.py --demo                        # print sample pairs
python generate_pairs.py --hf --max-docs 20000 --extra data/seed_conversational.txt --out-dir data/
python train/train_byt5.py --out models/byt5-mn --max-steps 5000 --resume
python eval/evaluate.py --model models/byt5-mn --model-name byt5-small   # baselines run without --model
python eval/error_analysis.py results/pred_real_test.csv --system byt5-small
python tools/real_test.py check
python rag/corpus.py && python eval/retrieval_eval.py  # mn Wikipedia passages; retrieval with Latin vs transliterated queries
python eval/llm_translit.py --dry-run                  # Claude zero-shot / random / retrieved few-shot (needs ANTHROPIC_API_KEY)
python tools/chat_data.py expand|pairs|test|mix        # chat-style stage-3 data: Gemma expands data/chat_seed_cyr.txt, Latin side via generate_pairs
python tools/prune_model.py --teacher models/byt5-small-chat2 --out models/byt5-pruned-4e2d-1024 --enc-layers 0,3,7,11 --dec-layers 0,2 --d-ff 1024
python tools/phone_bench.py --model models/byt5-pruned-4e2d-1024-ft2   # CPU one-message-at-a-time CER/latency, fp32 vs int8
python tools/export_onnx.py --check && python tools/word_dict.py && python tools/pack_keyboard_assets.py   # Android keyboard assets
python tools/context_eval.py                           # word accuracy alone vs with 2 previous Latin words (keyboard)
python tools/xai_attention.py --text "za bi margaash ugluu irne" --errors results/chat2/pred_real_test.csv   # heat map + error categories
python tools/tagger_data.py && python train/train_tagger.py && python tools/export_tagger_onnx.py --check  # encoder-only tagger
python train/train_lstm_tagger.py && python tools/export_tagger_onnx.py --lstm --model models/lstm-tagger --out models/onnx/lstm-tagger --check  # BiLSTM tagger: 8.5 MB int8, beats the ByT5 tagger
MODEL_BACKEND=onnx-tagger MODEL_DIR=models/onnx/tagger-4e-full uvicorn app.main:app   # torch-free serving (deploy/onnx.Dockerfile)
MODEL_DIR=models/byt5-mn uvicorn app.main:app --port 7860   # /keyboard = web keyboard mode (POST /suggest)
python tools/build_site.py                             # static browser demo in site/ (GitHub Pages); open site/selftest.html over http, needs bad=0
powershell -File tools/start_emulator.ps1               # Android emulator with the IME installed (portable JDK/Gradle/SDK in ~/tools)
python tools/youtube_comments.py && python annotate/db.py load data/raw/youtube_candidates.jsonl
uvicorn annotate.main:app --port 8000                  # or: docker compose up annotate
python annotate/db.py export && python tools/real_test.py import data/real_test_todo.csv
```

Scripts are run from the repo root as files (`python eval/evaluate.py`), not with `-m`; each
one puts the repo root on `sys.path` itself.

## Layout

- `generate_pairs.py`: Cyrillic corpus → JSONL `{"src", "tgt", "style"}`. Keep this format.
  Style mappings live in the `MNS` and `INFORMAL` dicts. Splits are by hash of `sentence_key()`.
- `baselines/`: `GreedyRuleTransliterator` (rules.py) and two tables. Interface: `.transliterate(str)`, `.batch(list)`.
- `translit/model.py`: `Seq2SeqTransliterator`, shared by eval and app (segmenting, batching, beam 4).
- `eval/metrics.py`: all metric definitions; `eval/evaluate.py` and `train/train_byt5.py` import from it.
- `tools/real_test.py`: the only intended way to edit `data/real_test.jsonl`.
- `annotate/`: annotation UI (FastAPI + SQLite `data/annotate.db`, no torch) for real-test candidates;
  it exports CSV for `real_test.py import` and never writes `real_test.jsonl` itself.
- `rag/`: knowledge base (Mongolian Wikipedia → `data/rag/passages.jsonl`) and retrievers (BM25 via bm25s,
  dense via multilingual-e5-base; indexes cached in `data/rag/`). `eval/retrieval_eval.py` measures how close
  Latin / transliterated queries get to the hand-made Cyrillic query's top-k. Never index real_dev/real_test.
- `app/`: FastAPI (`POST /transliterate` → `{"model", "baseline", "candidates", "alternatives", ...}`,
  `POST /feedback` logs the reader's word pick to `data/feedback/`) + `static/index.html`.
- `site/`: public static demo (GitHub Pages, `gh-pages` branch). `site/engine.js` is a JS port of
  `translit/onnx_backend.py`, `mixed.py` and `keyboard.py` on onnxruntime-web; keep it in sync with them and
  re-run `site/selftest.html` (compares with Python outputs from `tools/build_site.py`). `site/assets/` is git-ignored.
- `translit/candidates.py`: beam n-best → per-word alternatives for the pick-a-word UI and the keyboard.
- `keyboard/android/`: pinyin-style Android IME prototype (Kotlin + ONNX Runtime); assets come from
  `tools/export_onnx.py` (pruned student → encoder/decoder int8 ONNX) and `tools/word_dict.py`, never committed.
- On-device model: `tools/prune_model.py` keeps a subset of blocks / FF neurons of the fine-tuned teacher.
  Training a ByT5 from scratch did not learn here (see PROGRESS.md 2026-10-16); prune instead.
- `notebooks/train_byt5.ipynb`: Colab notebook; it calls the scripts above, so keep CLI flags in sync.

## Rules

- **`data/real_test.jsonl` is evaluation-only.** Never use it for training, validation or
  tuning, and never pre-fill its references with model output (baselines only).
  `generate_pairs.py` excludes its sentences; `train_byt5.py` refuses files named `real_test`.
  `data/chat_test.jsonl` (the dummy chats in `data/raw/dummy_chat_*.json`) is evaluation-only too; `tools/chat_data.py pairs` drops its sentences.
- Don't change metric definitions without saying so: results are compared across runs in PROGRESS.md.
- Precision: ByT5/T5 gives NaN in fp16. Use bf16 on Ampere or newer, fp32 on T4 (`pick_precision`).
- Length budget: ByT5 works on bytes and Cyrillic is 2 bytes per letter. Generated src and tgt
  stay ≤ 254 bytes (`--max-chars 110`); the model uses max length 256.
- `transformers` API drift (v4 vs v5): the train script checks for `eval_strategy` vs
  `evaluation_strategy`, `processing_class` vs `tokenizer`, `warmup_ratio` vs `warmup_steps` (float ratio)
  and `group_by_length` vs `train_sampling_strategy` itself. Keep those checks, and treat its
  "WARNING: ... ignored" line as a bug to fix: runs before 2026-10-10 (v1–v3) silently had no warmup.
- Inference must work on CPU. The app must still run (baseline only) when no model is available.
- UI text is in Mongolian Cyrillic. Code, comments and docs are in English.
- Read text files with `encoding="utf-8-sig"` when they may come from Windows (BOM).
- Generated data, `models/` and `results/pred_*.csv` are git-ignored; don't commit them.
  `.gitattributes` forces LF line endings.
- Python 3.10–3.12 for training and the app (torch/transformers support).
