# mn-translit — informal Latin-script → Cyrillic Mongolian

Many Mongolians type in Latin letters with no fixed convention, often mixed with Cyrillic and English
in the same message. Search, chatbots, machine translation and screen readers only understand the
Cyrillic. This project converts the Latin back to correct Cyrillic, on a server and inside a phone keyboard.

```
za bi margaash ugluu irne      →  За би маргааш өглөө ирнэ
bi meeting-ruu orloo           →  Би meeting-рүү орлоо
ene kino үнэхээр goy bna       →  Энэ кино үнэхээр гоё байна
```

**Live demo: [bayanmunkhe.github.io/mongolian-translit](https://bayanmunkhe.github.io/mongolian-translit/)**
(message converter and a pinyin-style keyboard page; the 19 MB model runs in your browser, nothing is sent to a server)

<p align="center"><img src="results/keyboard/emulator_typing.gif" width="420"
  alt="Android keyboard prototype: typing 'Yag odoo ene prototype-g turshij bna' produces 'Яг одоо энэ prototype-г туршиж байна'"></p>

**Results** on 195 real YouTube comments that were never used for training or tuning
(character error rate, lower is better; `results/chat2/results.md`, `results/llm/results.md`):

| System | Size | Real-test CER | Char. accuracy |
|---|---:|---:|---:|
| Rule baseline (naive letter map) | – | 19.7 | 80% |
| Gemma 3 12B, zero-shot | 12B | 18.1 | 82% |
| Gemma 3 12B, 5 retrieved examples | 12B | 12.5 | 88% |
| **ByT5-small, fine-tuned in 3 stages** (server) | 300M | **6.2** | **94%** |
| Pruned ByT5, 4 enc + 2 dec blocks, ONNX int8 | 46M | 7.7 | 92% |
| **BiLSTM byte tagger, ONNX int8** (keyboard) | 19 MB | 7.9 | 92% |

* The fine-tuned 300M model has half the errors of a 40× larger LLM given retrieved examples, at
  1/50–1/1000 of the serving cost, and runs offline on a CPU.
* The keyboard model converts a word in **0.3 ms** on a desktop CPU and gets 90% of 800 real-test words
  right with 2 words of context (`results/keyboard/`); the keyboard adds a dictionary and the user's own
  picks on top.
* Differences are checked with bootstrap confidence intervals (`results/*/bootstrap_*.md`).

**What is in the repo:** synthetic data generation (1M pairs with sampled writing styles), a
hand-annotated real test set and annotation UI, ByT5 fine-tuning (Colab T4 or local GPU), evaluation
with rule and LLM baselines, structured pruning and ONNX export, a FastAPI web demo with per-word
alternatives and a pinyin-style keyboard page, and a Kotlin Android IME running the 19 MB model
on-device. Built as the AI Engineer capstone (AI Academy Asia, NLP track); the full experiment log
is in `PROGRESS.md`.

## Repo layout

```
generate_pairs.py            synthetic data: Cyrillic corpus -> (informal Latin, Cyrillic) JSONL
data/seed_conversational.txt hand-written chat sentences (oversampled; reduces the Wikipedia domain gap)
data/real_test.jsonl         YOUR hand-collected real test set (evaluation only, tracked in git)
tools/real_test.py           build / import / check the real test set
baselines/                   mns5217.py (reverse MNS 5217:2012), naive.py (u→у, v→ү, h→х, o→о …)
translit/model.py            inference wrapper (batching, long-text splitting, beam search)
train/train_byt5.py          Seq2SeqTrainer fine-tuning (early stopping on valid CER)
notebooks/train_byt5.ipynb   Colab T4 notebook: data → train → evaluate → push to Hub
eval/metrics.py              CER, WER, exact match, ambiguous-letter accuracy, English preservation
eval/evaluate.py             all systems × (synthetic test, real test) → markdown table + CSV
eval/error_analysis.py       most frequent error types of one system
app/                         FastAPI backend + Mongolian web UI (n-best candidates, per-word alternatives, /feedback)
translit/candidates.py       n-best / per-word alternatives for the UI; translit/onnx_backend.py torch-free tagger backend
tools/chat_data.py           chat-style stage-3 data (hand seeds -> Gemma expansion -> Latin variants)
tools/prune_model.py         structured pruning of byt5-small for on-device use (4 enc + 2 dec blocks, d_ff 1024)
tools/export_onnx.py         seq2seq -> ONNX int8 (encoder + decoder); tools/phone_bench.py latency per word
tools/tagger_data.py         byte -> output-string labels for the encoder-only tagger; train/train_tagger.py trains it
tools/export_tagger_onnx.py  tagger -> ONNX int8 (argmax in graph) + CER/latency check
tools/context_eval.py        word accuracy with 0/1/2 previous words of context (keyboard setting)
tools/word_dict.py           keyboard dictionary (50k Latin words -> top-3 Cyrillic); tools/word_bigrams.py next-word counts
tools/pack_keyboard_assets.py copies ONNX models + dictionary + bigrams into keyboard/android/.../assets
keyboard/android/            Android IME prototype (Kotlin, onnxruntime-android): pinyin-style suggestion strip
tools/xai_attention.py       cross-attention heat maps for the XAI slide
rag/, eval/retrieval_eval.py Mongolian Wikipedia knowledge base; retrieval with Latin vs converted queries
annotate/                    annotation UI for real-test candidates (FastAPI + SQLite)
Dockerfile, deploy/          CPU containers: full torch app, and deploy/onnx.Dockerfile (torch-free, ~300 MB)
.github/workflows/tests.yml  pytest on every push
tests/                       pytest for everything that doesn't need torch
```

## Setup (macOS / Linux / Windows)

Use Python **3.10–3.12** (best support in torch/transformers).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # training/eval; on CPU-only machines this installs CPU torch
pip install -r requirements-app.txt
pytest -q
```

Data, models and results are git-ignored and get re-created by the scripts. The only
data files in git are the hand-made ones: `data/seed_conversational.txt`, `data/chat_seed*_cyr.txt`,
`data/real_{train,dev,test}.jsonl`, `data/chat_test.jsonl` and `data/raw/dummy_chat_*.json`. Put
models on Google Drive or the Hugging Face Hub.

## 1. Data

```bash
python generate_pairs.py --demo                        # see what the styles look like
python generate_pairs.py --hf --max-docs 20000 \
    --extra data/seed_conversational.txt \
    --max-train-pairs 200000 --out-dir data/
```

* Each sentence gets its own sampled **writing style**: a letter map for ө/ү/х/ц/ж/в/е/й/ы/ь/ъ/ю/я
  (10% of sentences use strict MNS 5217), lowercasing, long-vowel collapsing, punctuation
  stripping, typos, and a little per-letter inconsistency. The style is stored in
  the `style` field.
* 25% of sentences get 1–2 English words, which stay unchanged in src and tgt and
  sometimes carry a case suffix (`meeting-ruu` → `meeting-руу`).
* Splits are made **by sentence** using a hash of the normalized text, so a sentence
  never appears in two splits. Sentences from `data/real_test.jsonl` are excluded
  from every split.
* Sentences longer than 110 characters are cut into chunks, so src and tgt stay under
  254 bytes (ByT5 works on bytes and Cyrillic takes 2 bytes per letter).
* Other corpora: `--input file.txt …`, or a different HF dataset with
  `--hf-dataset <name> --hf-config <cfg> --text-field text` (e.g. CC-100 / OSCAR mn).

## 2. Real test set (150–200 messages)

Collect real Latin-script messages (your own chats, or ones people give you permission to
use; anonymize names and phone numbers). Keep them **exactly** as written, typos included.

```bash
# one message per line in raw.txt
python tools/real_test.py template raw.txt     # -> data/real_test_todo.csv (tgt pre-filled by naive rules)
# correct every tgt cell in Excel / Google Sheets / Numbers and save as CSV UTF-8
python tools/real_test.py import data/real_test_todo.csv
python tools/real_test.py check                # validation, stats, leakage vs synthetic data
python tools/real_test.py add                  # or: enter items one by one
```

Pre-fills come from a rule baseline and never from the model, so the references can't be
biased toward the system being evaluated. Write the Cyrillic the way the writer *meant*
it, with proper capitalization and punctuation, and leave English words unchanged. Rows
can have optional `source`/`note` columns. After adding items, re-run `generate_pairs.py`
so the exclusion is applied.

## 3. Training (Colab T4)

Open `notebooks/train_byt5.ipynb` in Colab and run all cells. Data, checkpoints and
results go to Google Drive, and training resumes automatically after a disconnect.
Locally or on any GPU:

```bash
python train/train_byt5.py --train data/train.jsonl --valid data/valid.jsonl \
    --out models/byt5-mn --max-steps 5000 --resume
```

Defaults: lr 3e-4 (AdamW, linear decay, 5% warmup), batch 16 × grad-accum 2, max
length 256 bytes, evaluation every 500 steps on 500 validation examples (greedy decoding),
early stopping on valid CER (patience 4), best checkpoint kept, seed 42.
Precision: **bf16** on Ampere or newer GPUs and **fp32 on T4**, because T5-family models
overflow to NaN in fp16 (`--fp16` forces it anyway). For comparison: `--model google/mt5-small`.

The final model is trained in three stages, each continuing from the previous one:

1. **synthetic** (1M pairs from Wikipedia) → `byt5-mn`;
2. **real_train × 20 + 10k synthetic** (hand-corrected chat messages, see `data/real_train.jsonl`) → `byt5-small-v4-real`;
3. **chat mix**: `tools/chat_data.py expand` turns the hand-written seeds `data/chat_seed_cyr.txt` /
   `chat_seed2_cyr.txt` into ~2.6k chat sentences with a local open model (Gemma 3 via Ollama), `pairs` makes 6
   Latin variants per sentence (abbreviations, split suffixes, particles), `mix` joins them with real_train × 20
   and 10k synthetic → `byt5-small-chat2` (demo default). `data/chat_test.jsonl` (40 dummy chat messages) is
   evaluation-only and is dropped from the pairs.

## 4. Evaluation

```bash
python eval/evaluate.py --model models/byt5-mn --model-name byt5-small
python eval/error_analysis.py results/pred_real_test.csv --system byt5-small
```

Every system is run on `data/test.jsonl` (synthetic) and `data/real_test.jsonl` (real). Reported metrics:
CER, WER, sentence exact match, the same three case-insensitive, accuracy on ambiguous
letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ, found through character alignment), and the share of
English/Latin words preserved. Outputs: `results/results.md`, `results/metrics.json`,
`results/pred_*.csv` (per-example predictions and CER for every system).

Additional metrics: chrF and BLEU (sacrebleu), and `tools/context_eval.py` reports word accuracy when the
word is converted alone vs. with its 1–2 previous words (the keyboard setting: 75% → 87% with context).

## 5. Web demo

```bash
uvicorn app.main:app --port 7860     # loads models/byt5-small-chat2 (else older ones); MODEL_DIR=... overrides
# open http://localhost:7860
MODEL_BACKEND=onnx-tagger MODEL_DIR=models/onnx/tagger-4e-full uvicorn app.main:app   # torch-free, ~2 ms/message
```

`POST /transliterate {"text": "..."}` → `{"model": "...", "baseline": "...", "candidates": [...], "alternatives": [...]}`:
the model answer, up to 3 whole-sentence candidates with their probabilities (beam n-best, casing-merged), and
per-word alternatives the UI shows as drop-downs. `POST /feedback` logs the user's pick to
`data/feedback/choices.jsonl` (never used for training automatically).

**Keyboard mode** (`/keyboard`, works on any phone browser, iPhone included): a pinyin-style strip for the word
being typed. `POST /suggest {"word", "context": [prev Latin words], "prev_cyr"}` returns candidates from the
dictionary (re-ranked by bigrams), the context-aware ONNX tagger (`KEYBOARD_TAGGER`, default the best of
`models/onnx/lstm-tagger-512x3-ft`, `tagger-4e-full`, …) and the raw Latin last; picks that were not the first
suggestion are remembered in the browser (`localStorage`). Logic in `translit/keyboard.py`.
`POST /search {"text": "..."}` searches Mongolian Wikipedia with the raw Latin text and with its conversion
(needs `data/rag/` from `rag/corpus.py`; the first start downloads the e5 model, ~1 GB, once;
`RAG_RETRIEVER=bm25` works offline). Inference runs on CPU. Without a model the app serves the baseline only.

**Hosting options** (free Spaces need a PRO account as of 2026-10; a quick Cloudflare tunnel,
`tools/start_public_demo.ps1`, serves the local app for a presentation):

* `site/` — the public demo: a static page that runs the BiLSTM tagger in the browser with onnxruntime-web
  (`site/engine.js` ports `translit/onnx_backend.py`, `mixed.py` and `keyboard.py`). `python tools/build_site.py`
  writes `site/assets/` (model, trimmed dictionary and bigrams, Python reference outputs); `site/selftest.html`
  must report `bad=0` before deploying. The built folder is published on the `gh-pages` branch (GitHub Pages).
* `Dockerfile` — full app with torch and `byt5-small-chat2` (~2 GB image, beam search, n-best).
* `deploy/onnx.Dockerfile` — `MODEL_BACKEND=onnx-tagger`: onnxruntime only, ~300 MB image, runs on the
  smallest free tiers (Render, Cloud Run, Fly). Slightly lower accuracy (tagger, no beam search).
* `gradio_app.py` + `tools/push_space.py` — Gradio Space (`deploy/hf_space_README.md`, `deploy/space_requirements.txt`).

## 6. On-device: pruning, ONNX and the keyboard

A ByT5 trained from scratch at phone size did not learn (see PROGRESS.md), so the on-device models are
**pruned** from the fine-tuned byt5-small instead:

```bash
python tools/prune_model.py --teacher models/byt5-small-chat --enc-layers 0,3,7,11 --dec-layers 0,2 --d-ff 1024 --out models/byt5-pruned-4e2d-1024
python train/train_byt5.py --model models/byt5-pruned-4e2d-1024 --train data/student_train.jsonl ... --out models/byt5-pruned-4e2d-1024-ft2
python tools/export_onnx.py --model models/byt5-pruned-4e2d-1024-ft2 --out models/onnx/pruned-4e2d-1024-ft2 --check   # int8: 28 + 20 MB
python tools/phone_bench.py --model models/byt5-pruned-4e2d-1024-ft2                                                  # ms per word, p50/p95/p99
```

For the keyboard the seq2seq decoder is still the slow part, so a second model converts in **one pass**:
an **encoder-only tagger** that labels every input byte with the output string it becomes
(edit-alignment labels, 4000-label vocabulary):

```bash
python tools/tagger_data.py --max-train 1100000 --vocab 4000                 # data/tagger_full/
python train/train_tagger.py --encoder models/byt5-pruned-4e2d-1024-ft2 --train data/tagger_full/train.jsonl \
    --valid data/tagger_full/valid.jsonl --vocab data/tagger_full/vocab.json --out models/tagger-4e-full --steps 40000
python tools/export_tagger_onnx.py --model models/tagger-4e-full --out models/onnx/tagger-4e-full --check   # int8 34 MB, ~1 ms/message
python tools/context_eval.py --onnx-tagger models/onnx/tagger-4e-full
python train/train_lstm_tagger.py --out models/lstm-tagger                   # BiLSTM on the same labels (no pretraining)
python tools/export_tagger_onnx.py --lstm --model models/lstm-tagger --out models/onnx/lstm-tagger --check   # int8 8.5 MB
python tools/word_dict.py && python tools/word_bigrams.py                    # data/keyboard/{word_dict,bigrams}.json
python tools/pack_keyboard_assets.py --tagger models/onnx/lstm-tagger-512x3-ft  # -> keyboard/android/app/src/main/assets
```

Surprise: on these labels a byte-level BiLSTM trained from scratch beats the pretrained pruned encoder: 2 layers ×
384 (8.7M, 10 minutes) gives real test CER 8.52 vs 8.77 at a quarter of the size, and 3 layers × 512 (19.9M,
`--hidden 512 --layers 3 --steps 30000`, then 30k more with `--init ... --lr 5e-4`) reaches **7.9** (dev 4.30),
on par with the pruned seq2seq without a decode loop: int8 19.3 MB, 0.3 ms per word, 90% standalone word
accuracy. That is the keyboard's default tagger (`models/onnx/lstm-tagger-512x3-ft`);
both kinds export to the same ONNX interface (`input_ids`, `attention_mask` → `label_ids`).

The Android IME (`keyboard/android/`, open in Android Studio) works like a pinyin keyboard: the user types
Latin, the strip shows Cyrillic candidates, space commits the first one. Lookup order: the user's own past
picks → packed dictionary (30k words, re-ranked by the previous committed word's bigram counts) → tagger with
the 2 previous words as context → seq2seq model only when the tagger meets a byte it has no label for. The
raw Latin is always the last candidate, so English stays English.

## Development notes

`.gitattributes` keeps line endings as LF on Windows and macOS. `.github/workflows/tests.yml` runs `pytest -q`
(no torch) on every push.

## Limitations

* The training data is synthetic and mostly from Wikipedia, so slang and chat style are
  under-represented. The real test set measures this gap.
* Capital letters can't be recovered reliably from lowercase input (proper nouns),
  which is why case-insensitive scores are reported too.
* Short English words that are also valid Mongolian Latin spellings (`bus`, `tor`) are ambiguous.
* Data: Mongolian Wikipedia, CC BY-SA 4.0.
