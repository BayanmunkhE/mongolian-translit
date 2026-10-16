# mn-translit — informal Latin-script → Cyrillic Mongolian

AI Engineer capstone (AI Academy Asia, NLP track).

Many Mongolians type in Latin letters with no fixed convention:

```
za bi margaash ugluu irne      →  За би маргааш өглөө ирнэ
bi meeting-ruu orloo           →  Би meeting-рүү орлоо
```

ө is written u/o/ö, ү as u/v/ü, х as h/kh/x, long vowels get dropped, punctuation
is skipped, and English words are mixed in. This project fine-tunes **ByT5-small**
on synthetic pairs to convert such text to correct Cyrillic. The model is compared
with two rule baselines on a synthetic test set and on a hand-collected real test
set, and served in a web demo.

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
app/                         FastAPI backend + Mongolian web UI
Dockerfile, deploy/          CPU container for Hugging Face Spaces
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
data files in git are `data/seed_conversational.txt` and `data/real_test.jsonl`. Put
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

## 5. Web demo

```bash
uvicorn app.main:app --port 7860     # loads models/byt5-small-v4-real (else byt5-mn); MODEL_DIR=... overrides
# open http://localhost:7860
```

`POST /transliterate {"text": "..."}` → `{"model": "...", "baseline": "...", ...}`.
`POST /search {"text": "..."}` searches Mongolian Wikipedia with the raw Latin text and with its conversion
(needs `data/rag/` from `rag/corpus.py`; the first start downloads the e5 model, ~1 GB, once;
`RAG_RETRIEVER=bm25` works offline). Inference runs on CPU. Without a model the app serves the baseline only.

**MacBook from the zip bundle** (`mn-translit-mac.zip`: code, hand-made data, results, RAG index, small model):

```bash
cd ~/Desktop && unzip mn-translit-mac.zip && cd mn-translit
brew install python@3.12                       # if python3.12 is missing
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-app.txt pytest
pytest -q && uvicorn app.main:app --port 7860  # then open http://localhost:7860
```

Start it once at home with internet before presenting, so the e5 model is cached.

**Hugging Face Spaces:** upload the model to the Hub (last cells of the notebook), then
create a Docker Space and push `Dockerfile`, `requirements-app.txt`, `app/`, `baselines/`
and `translit/`. Use `deploy/hf_space_README.md` as the Space's `README.md`, and set the Space
variable `MODEL_ID=<user>/<model-repo>`.

## Git: Windows ↔ MacBook

```bash
# on the machine that has the folder
git init && git add . && git commit -m "Initial project skeleton"
# create an empty repo on GitHub, then:
git remote add origin https://github.com/<you>/mn-translit.git
git branch -M main && git push -u origin main
# on the MacBook
git clone https://github.com/<you>/mn-translit.git
```

`.gitattributes` keeps line endings as LF on both machines.

## Limitations

* The training data is synthetic and mostly from Wikipedia, so slang and chat style are
  under-represented. The real test set measures this gap.
* Capital letters can't be recovered reliably from lowercase input (proper nouns),
  which is why case-insensitive scores are reported too.
* Short English words that are also valid Mongolian Latin spellings (`bus`, `tor`) are ambiguous.
* Data: Mongolian Wikipedia, CC BY-SA 4.0.
