# Capstone progress (14 days)

Tick boxes as you go (`[x]`). GitHub and VS Code's Markdown preview render this as a checklist.

## Done: code skeleton (all tested locally except real GPU training)
- [x] `generate_pairs.py`: per-sentence writing styles, English insertion, sentence-level split, real-test exclusion
- [x] `data/seed_conversational.txt`: 80 chat sentences (**please proof-read**)
- [x] Baselines: `baselines/mns5217.py` (reverse MNS 5217:2012), `baselines/naive.py`
- [x] Metrics + `eval/evaluate.py` (markdown table, CSV predictions) + `eval/error_analysis.py`
- [x] `tools/real_test.py`: template / import / add / check (validation + leakage check)
- [x] `train/train_byt5.py`: smoke-tested on a tiny model with transformers 5.18 (train → eval → best checkpoint → save)
- [x] `notebooks/train_byt5.ipynb` for Colab T4 (Drive storage, auto-resume)
- [x] FastAPI app + Mongolian UI; works with and without a model
- [x] Dockerfile + HF Spaces README, `.gitignore`, `.gitattributes`, pytest (6 tests)

## Status 2026-10-12 (preliminary defense on 2026-10-13)
- **Best model:** `models/byt5-small-v4-real` (real test CER 6.59; base-v4-real 6.19 is a tie and 2x slower on CPU).
  The demo loads it by default and has a Wikipedia search card (`POST /search`).
- **Direction:** the project is framed as the front-end of a RAG system: retrieval results are in the log below.
- **real_dev fix:** two references had дүү for дуу ("chi duu sonsoj", "oor duu oor clip"); fixed in
  `data/real_dev.jsonl` and `data/annotate_train.db`. real_dev numbers above were scored before (2 characters).
- **Mixed-script input (found 2026-10-12):** the model corrupts text that is already Cyrillic ("Би маргааш өглөө
  ирнэ" → "Би мааргааных шиглээ ирнээ"); it only ever saw Latin input. The app now uses `translit/mixed.py`:
  Cyrillic words are kept and only Latin runs go to the model (pure-Latin messages unchanged, so no metric moves;
  real_train/dev/test contain no Cyrillic in src). Not covered by any test set yet: a mixed-script eval set is open.
- **Public demo without HF:** HF now requires PRO ($9/month) for Docker Spaces (402 on create). Running instead on
  this PC (GPU) + Cloudflare quick tunnel: `tools/start_public_demo.ps1` (link changes on each start). 50 parallel
  requests: 0 errors, last one waits ~28 s (requests are serialized); micro-batching is the next fix.
- **Disk cleanup (user-approved):** all `checkpoint-*` dirs deleted (final weights kept, so no resuming old runs),
  models `byt5-mn` (v1), `byt5-mn-v2`, `byt5-scratch-v3*` and `data/v2`, `data/v3` deleted; their numbers stay above.
- **Deploy:** `tools/push_space.py` uploads a public HF Space (Docker, `deploy/space.Dockerfile`, model + e5 index baked in).
- **Preliminary defense done (2026-10-13).**
  - **Deck:** 23 main slides plus appendix Н1–Н5.
    - Gemma claims softened.
    - Added: RAG diagram, ERD, EDA, 3-tier TCO, business model/ROI, backup demo slide, timed speaker notes.
  - **Demo UI** (`app/static/index.html`):
    - highlights words that differ from the rule baseline, and Latin words kept as written;
    - groups the example buttons;
    - one button converts and searches.
  - **Gaps against the course guidance** (AI_Engineer_Capstone_Project_Guidance.pptx, 13-slide structure) still open for the final defense:
    - XAI (cross-attention heatmap or error categories);
    - an LSTM/RNN baseline (the NLP track lists TF-IDF / LSTM / fine-tuned Transformer);
    - BLEU/chrF/F-score next to CER;
    - P50/P99 latency;
    - cloud hosting instead of the PC tunnel;
    - CI/CD and a GitHub repo.
  - **To restart the demo:** run `tools/start_public_demo.ps1`. It prints a new link, so regenerate the QR on the demo slide.
  - **Mac zip:** `Desktop\mn-translit-mac.zip` predates the index.html change.
- **On-device plan, step 1 done (2026-10-15, `tools/phone_bench.py`, `results/phone/bench.md`):** small-chat on CPU
  (4 threads, one message at a time). Greedy costs nothing vs beam 4 (real 6.55 vs 6.53; chat 6.36 vs 6.58). Dynamic
  int8 costs ~0.4 CER (6.91 / 6.80) and gives 2.4x speed: p50 443 ms real / 253 ms chat, 288 MB. A phone is ~3-5x
  slower and an iOS keyboard extension has ~50-70 MB, so a 10-20x smaller student is required (step 2:
  distillation into a 15-30M byte-level model with a 2-layer decoder, or an encoder-only character tagger).
  Latency is dominated by decode steps (2 bytes per Cyrillic char), so fewer decoder layers matter most.
- **Step 2, round 1 (from scratch) FAILED:** 29.6M ByT5 (d_model 512, 6 enc / 2 dec, d_ff 1536) trained from
  scratch on the gold mix (`data/student_train.jsonl` = v4 1M + chat x3 + real_train x20) stalled at train loss
  1.49 = a decoder-only byte LM ignoring the encoder (dev CER 77%, then runaway). Same failure as the 300M
  from-scratch run. Do not train ByT5 from scratch on this task.
- **Step 2, round 2 (prune the teacher) works but is not there yet:** `tools/prune_model.py` keeps encoder blocks
  0,3,7,11, decoder block 0 and the 768 most active FF neurons per block of small-chat (299.6M → 31.1M), then
  fine-tunes 12k steps at lr 3e-4 (`results/queue_prune.py`, best ckpt dev CER 7.7%). Greedy, one message at a
  time: real test **11.5** (teacher 6.5), chat test 12.5 (6.4), 119 MB fp32, p50 318 ms real / 99 ms chat on the
  desktop. Problems: some runaway repetitions (batched eval real test 15.0, p95 beam 5.2 s), and dynamic int8
  wrecks it (real 37.9) unlike the teacher (−0.4). Round-3 candidates: 2 decoder blocks, d_ff 1024, ≥1.5 epochs,
  an n-gram repeat guard at inference, int8 that leaves lm_head/embeddings in fp32 (or QAT).
- **Step 2, round 3 (`models/byt5-pruned-4e2d-1024-ft`, 45.8M: enc 0,3,7,11 / dec 0,2 / d_ff 1024, 20k steps,
  dev CER 5.6%):** one message at a time, greedy: real test **9.0** fp32 / **9.2 int8** (teacher 6.5), chat test
  **9.1 / 9.1**; int8 (encoder+decoder Linear only, lm_head fp32) no longer hurts; 175 MB fp32, 50 MB int8.
  Batched eval real test 10.5 (CI 7.6–14.2): a few runaway outputs remain. Latency is the problem: per message
  p50 526 ms real / 198 ms chat (int8, desktop 4 threads) because d_model stays 1472; per single word (keyboard
  use) p50 55 ms (teacher 90 ms) → roughly 150–250 ms on a phone. Accuracy goal (≤9) met at the margin, size
  (≤30 MB) and per-message latency not. Options next: width reduction (low-rank/SVD of the 1472-wide matrices),
  an encoder-only tagger (no decode loop), or ship per-word conversion with a dictionary cache in front.
  (Its full eval was rerun overnight: real_dev 7.02, chat_test 9.10, batched real test 10.54.)
- **Overnight 2026-10-16 (`results/queue_night.py`):**
  - **small-chat2** (stage 3 retrained with `data/chat_seed2_cyr.txt`, 15,942 chat pairs): real test **6.15**
    (small-chat 6.53), dummy chat **5.26** / EM 72.5% (6.58 / 55%), real_dev 4.52 (tie). Gaps' CIs include 0
    (−0.06…+0.89 real, −0.41…+3.23 chat) but nothing regressed → new demo default candidate. `results/chat2/`.
  - **pruned round 3 continued** (`byt5-pruned-4e2d-1024-ft2`: +20k steps at lr 1e-4 on the refreshed mix): one
    message at a time real test **7.72** fp32 / **7.89** int8 (was 9.0 / 9.2), chat **7.46 / 7.35**, real_dev 6.41,
    p50 349 ms real / 145 ms chat int8, 50 MB. Best on-device candidate so far.
  - **Copy-pretraining experiment failed, and the reason explains every from-scratch failure:** even the src→src
    copy task did not learn (loss 192 → 4.1 plateau, CER 480%). Diagnosis: HF initialises an untied T5 `lm_head`
    with std 1.0, so initial logits are ~N(0, d_model) and the first loss is 211–295 instead of ln(384)=5.95; the
    model then settles in a decoder-only plateau. Fixed in `train_byt5.py` (`--from-scratch` rescales lm_head to
    d_model^-0.5 → initial loss 6.6). Precise cause: transformers 5.18 reports `tie_word_embeddings=True` for
    google/byt5-small and `from_config` ties lm_head to the std-1.0 embedding (the released checkpoint is untied);
    `train_byt5.py --from-scratch` now unties, asserts it, and rescales lm_head. (v3 = first fix attempt, skipped
    by its own guard; v4 = fix applied: first loss 7.9 instead of 281.)
  - **But the init was not the whole story.** `byt5-student-30m-v4` (fixed init, 25k steps) still plateaued at
    train loss 1.12 / real CER 71%. A tiny from_config T5 (2 layers, d_model 128) does learn byte copying in 2000
    steps (plain loop, lr 1e-3, no dropout), so from_config models are sound; the 30M run on the full data is an
    optimisation problem. Confirmed 06:20: the 30M config fails even the copy task (4000 Trainer steps CER 520%;
    plain-loop variants fp32/bf16 × lr 1e-3/3e-4 × dropout 0/0.1 all stay at loss 1.6–2.5 after 1500 steps,
    babbling), while a 2-layer d_model-128 model copies after 2000 steps. Deeper T5 stacks from scratch do not
    find the alignment here. **From-scratch ByT5 students are shelved**; the way to a narrow fast model is
    width reduction of the pruned teacher (SVD) or an encoder-only tagger.
  - **Round 4 (06:25–07:27):** the same pruning (4 enc / 2 dec / d_ff 1024) applied to the better teacher
    small-chat2 → `byt5-pruned2-4e2d-1024-ft`, 20k steps: real test 8.34 fp32 / 8.53 int8, chat 7.79, real_dev
    8.01 — slightly behind `byt5-pruned-4e2d-1024-ft2` (40k steps in total: 7.72 / 7.89 / 7.46 / 6.41), so the
    extra 20k steps mattered more than the better teacher. **Best on-device candidate: `byt5-pruned-4e2d-1024-ft2`.**
  - **Morning 2026-10-16, user said "do all of it":** demo now serves `small-chat2` (`DEFAULT_MODELS`), new
    tunnel + QR in the deck; failed from-scratch dirs deleted (4.6 GB); keyboard path (a) started:
    - `tools/export_onnx.py`: pruned-4e2d-1024-ft2 → encoder/decoder ONNX, weights moved from Constant nodes to
      initializers so ORT's dynamic quantizer works: **int8 28 + 20 MB, 8 ms/word** on the desktop (fp32 ONNX 18 ms,
      PyTorch 41 ms); agrees with PyTorch greedy on 117/120 words fp32, 109/120 int8.
    - `tools/word_dict.py`: 50k most frequent Latin words → up to 3 teacher candidates (4.6 MB; 2.4 MB packed
      top-30k). Context-free, so e.g. `shd` → Шийдье instead of шүү дээ: the strip + user dictionary mitigate.
    - `keyboard/android/`: Kotlin IME prototype (QWERTY grid, suggestion strip, user dictionary, ONNX greedy
      decode mirroring the Python harness). **Not built or run yet** (no Android SDK on this PC) — needs Android
      Studio + a phone; see its README.
- **Afternoon 2026-10-16 (user picked tracks 2, 3, 5):**
  - **Keyboard context (track 2):** converting a word with its two previous Latin words raises word accuracy from
    75% to 87% on both test sets (`tools/context_eval.py`, `results/keyboard/context_eval.md`); the IME now keeps
    the last two committed Latin words and strips the prefix's conversion (`Transliterator.convertInContext`).
  - **Rubric gaps (track 3):** chrF/BLEU added to `corpus_metrics` (small-chat2 real test chrF 89.3, BLEU 69.0;
    naive 53.7 / 15.7); phone_bench reports p99; `tools/xai_attention.py` draws cross-attention heat maps
    (clean monotonic diagonal, `results/xai/`) and an error-category table; git repo initialised (portable
    MinGit, first commits 91925ca, 1b07e54) with `.github/workflows/tests.yml`. **Cloud: HF now returns 402 for
    Gradio Spaces too** ("hosting Gradio and Docker Spaces on free cpu-basic requires PRO") — `gradio_app.py` +
    `tools/push_space.py` are ready for a PRO account; otherwise a free host must serve the 48 MB ONNX model.
  - **Encoder-only tagger (track 5):** `tools/tagger_data.py` (byte → output-string labels by edit alignment,
    4000-label vocab covers 99.99% of bytes), `train/train_tagger.py` (pruned 4-block encoder 27.7M + linear head,
    ~14 steps/s), `tools/export_tagger_onnx.py`. **Result (`models/tagger-4e`, best step 14k, early stop at 18k):**
    dev-mix CER 6.52 (pruned seq2seq 5.6, teacher 3.7); ONNX int8 **33.8 MB**, real test **CER 9.18** / chat
    **8.44** — on par with the pruned seq2seq's 9.0 / 9.1 of round 3, but **1–2 ms per message and 0.5 ms per
    word** on the desktop (seq2seq int8: 350 / 145 ms per message, 8 ms per word). The keyboard target
    (<10 ms/word, ≤ ~30 MB) is met with no decode loop; `results/phone/bench_tagger-4e_onnx.md`. Next: feed
    it more steps / the full 1M rows (it was trained on 400k), and use it as the keyboard's primary model with the
    seq2seq only for UNK labels.
    Broken dirs: `byt5-student-30m`, `-copy`, `-30m-v2`, `-30m-v3`, `-30m-v4` (deletable, ~0.5 GB each).
- **Overnight 2026-10-17 (`results/queue_tagger.py`, user asleep until 15:00):**
  - **Tagger on the full gold mix (`models/tagger-4e-full`, 1.05M rows, 40k steps, best step 34k, 54 min):**
    dev-mix CER **5.74** (400k-row tagger-4e: 6.52; pruned seq2seq 5.6). ONNX int8 (33.8 MB): real test
    **8.77** / EM 22.6 (tagger-4e 9.18), chat test **6.58** / EM 62.5 (8.44 / 45.0), still 2 ms/message and
    0.5 ms/word. With 2 previous words of context: word accuracy 83.6 → **87.0%** real, 82.0 → 86.1% chat
    (`results/keyboard/context_eval_tagger-4e-full.md`). Keyboard assets repacked with it (default of
    `pack_keyboard_assets.py`). Context still breaks laughter tokens (hha, hhe, haha) and `duu`→дөө: a repeat
    guard / dictionary hit covers those.
  - **Torch-free serving:** `translit/onnx_backend.py` + `MODEL_BACKEND=onnx-tagger` in the app, `deploy/onnx.Dockerfile`
    (onnxruntime only, ~300 MB image) — fits free hosting tiers where the 2 GB torch image does not.
  - **Keyboard bigrams:** `tools/word_bigrams.py` → `data/keyboard/bigrams.json` (37,936 previous words, 3.7 MB);
    `WordDictionary.kt` re-ranks dictionary candidates by the previous committed word; `MnLatinIme` passes it.
  - **Git hygiene:** `data/tagger/` (153 MB, generated) had been committed in 976938a; untracked in a8a7957 and
    ignored (`data/tagger*/`). The blob is still in history, so before the first GitHub push (100 MB limit) squash
    or rewrite those two commits (no remote yet, nothing pushed).
  - **LSTM baseline (rubric: TF-IDF / LSTM / Transformer):** `train/train_lstm_tagger.py`, a byte-level 2-layer
    BiLSTM tagger (8.7M params, no pretraining) on the same labels; queued after the tagger (`results/queue_lstm.py`).
    **Result (`models/lstm-tagger`, 20k steps, 10 min, best step 18k): dev-mix CER 5.21 — better than the pretrained
    pruned-ByT5 encoder tagger (5.74).** Real test **8.52** / EM 23.1 (ByT5 tagger 8.77), chat test 6.58 (tie).
    ONNX int8 (`--lstm` in `tools/export_tagger_onnx.py`, same inputs/outputs so the app backend and the keyboard
    load it unchanged): **8.5 MB** (ByT5 tagger 33.8), real 8.52 / chat 6.69, 0.4 ms/word, 2 ms/message. Standalone
    word accuracy 87.6% real / 91.0% chat (ByT5 tagger 83.6 / 82.0); the 2-word context adds little here (88.4 /
    89.3). Reading: for byte-level tagging with 1M labelled rows, pretraining does not pay at this size — the
    pretrained encoder wins only as a seq2seq (beam search, 6.15 CER). Candidate to replace the ByT5 tagger in the
    keyboard (4x smaller, same speed, slightly better). A 3-layer / hidden-512 variant (19.9M) is training
    (`models/lstm-tagger-512x3`, 30k steps, ~2 h). The process exits with 0xC0000409 at interpreter teardown after
    writing everything (CUDA/cuDNN on Windows); results are complete.
- **Demo: n-best word choice (2026-10-15).** `/transliterate` returns `candidates` (beam n-best, casing-only
  duplicates merged, real-odds softmax) and `alternatives` (word positions where candidates differ,
  `translit/candidates.py`); the page renders those words as a dropdown and POSTs the reader's pick to
  `/feedback` (`data/feedback/choices.jsonl`, git-ignored) — the future personalisation / training signal.
  The user's product goal is a pinyin-style keyboard: type Latin, pick the Cyrillic outcome.
- **Slides:** 17-slide Mongolian deck (incl. related work: MNS 5217, kirilleer.com, bichig.mn, SVM paper,
  Byambadorj et al. 2021 ACM TALLIP) with speaker notes (claude.ai artifact). Placeholders: name, demo link.
- **Open:** LLM comparison (`eval/llm_translit.py`, needs ANTHROPIC_API_KEY and a model choice); a Latin Q&A
  set for end-to-end RAG; deploy for ~50 users (HF Space or own domain + VPS); GitHub repo (git is not installed
  on the Windows PC; `.gitignore` drops `data/real_train.jsonl`/`real_dev.jsonl`, fix before the first commit);
  suffix convention for real-test items #58/104/114/118/186 and the u→и/m→м/k→к/Ѳ→Ө look-alike fixes.

## Plan
| Day | Task | Status |
|---|---|---|
| 1 | Put repo on GitHub, clone on MacBook, run `pytest` + `generate_pairs.py --demo` | [ ] |
| 1–4 | Collect 150–200 real Latin messages → `tools/real_test.py template/import/check` | [x] 195 YouTube comments via `annotate/` |
| 2 | Generate full data on Colab (`--hf --max-docs 20000 --extra …`), inspect 50 pairs by eye | [x] generated locally; inspection still to do |
| 2 | Run baselines on synthetic test → first results table | [x] |
| 3–4 | Train ByT5-small on Colab T4 (~5000 steps); check loss curve, valid CER | [x] trained locally (RTX 4070 SUPER), ~1.7 h |
| 5 | Evaluate all systems on synthetic + real test; error analysis | [x] |
| 6–7 | Improve the data based on errors (new styles, more conversational text), retrain | [ ] |
| 8 | Optional: mT5-small or char-level Transformer comparison | [ ] |
| 9 | Push model to HF Hub; deploy Docker Space; test the demo on phone + laptop | [ ] |
| 10–11 | Final evaluation (freeze real test!), tables, error examples, plots | [ ] |
| 12–13 | Report + slides: problem, data, method, results vs baselines, limitations | [ ] |
| 14 | Rehearse demo, buffer | [ ] |

## Results log
| Date | Run | Data | Steps | Synthetic CER | Real CER | Real EM | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-06 | mns5217 baseline | wiki20k+seed, 200k train pairs | – | 19.84 | 23.08 | 0.00 | real test = 195 YouTube comments |
| 2026-10-06 | naive baseline | wiki20k+seed, 200k train pairs | – | 17.76 | 19.33 | 1.03 | |
| 2026-10-06 | byt5-small v1 | wiki20k+seed, 200k train pairs | 5000 | 2.84 | 13.46 | 4.62 | local RTX 4070 SUPER, bf16, bs 16×2, lr 3e-4, beam 4; valid CER still falling at 5000. Real-test errors: chat abbreviations (bn, bga, sht), гоё→гой, added final "." |
| 2026-10-09 | byt5-small v2 | FineWeb-2 100k docs + seed, cleanup + chat style, 1M train pairs | 7000 | 3.68 (v2 test: 2.91) | 8.61 | 20.00 | resumed at 1000 with fused AdamW + 0 workers (~2x faster); valid CER flat at the end (3.00→2.99). Fixed: bn→байна, гоё. Left: added commas/stops, шд vs шүү дээ, English-in-Cyrillic refs (коммент, клип) |
| 2026-10-10 | byt5-small v3 | v2 data + no tgt punctuation when the writer typed none | 7000 | v3 test 2.14 | 8.34 | 24.10 | bs 8×4 (16×2 spilled into shared GPU memory because dwm held 2 GB). vs v2: −0.27 CER, 95% CI −0.79…+0.34 (tie) |
| 2026-10-10 | v3, beam 1 / 8 | decoding only | – | valid 2.18 / 2.14 (beam 4: 2.15) | 8.88 / 8.30 | 24.10 / 24.10 | all within noise; CPU median 881 / 840 / 921 ms per message for beam 1 / 4 / 8 (no speed gain from beam 1) |
| 2026-10-10 | byt5-small v3, lr 5e-4 | v3 data | 7000 | v3 test 2.00, valid 1.97 | 8.95 | 22.56 | better on synthetic valid, tie on real (vs v3: +0.61, CI −0.14…+1.71) |
| 2026-10-10 | ByT5-small from scratch | v3 data, random init | 7000 | 100 | 100 | 0 | diverged (loss ~27, grad norm 1e5, empty outputs); no warmup (see below) |
| 2026-10-10 | ByT5-small from scratch + warmup | v3 data, random init, warmup 350, length-grouped | 7000 | valid 100–158 | – | 0 | loss 742 → ~7.6 (still above uniform), then diverged at the end (loss 40, grad norm 1e5). Untuned (same lr/optimizer/bf16 as fine-tuning). Conclusion: at this budget, pretraining is what makes the task learnable |
| 2026-10-11 | byt5-small v3 + warmup | v3 data, warmup 5% + length-grouped (bug fixed) | 7000 | v3 test 2.18, valid 2.15 | 7.61 | 22.56 | 54 min. Tie with v3 (−0.22, CI −0.71…+0.26): the missing warmup did not hurt fine-tuning |
| 2026-10-11 | byt5-small v3, Adafactor | v3 data, warmup | 7000 | v3 test 2.12, valid 2.23 | 7.52 | 24.10 | 61 min. Tie with AdamW: the optimizer is not what makes base better |
| 2026-10-11 | **byt5-base** | v3 data, Adafactor, warmup | 7000 | v3 test **1.81**, valid 1.88 | **7.40** | 23.59 | 100 min, best ckpt 6000. AdamW state (580M params) spills out of 12 GB (4–5 s/it even at bs 4×8), so Adafactor at bs 8×4 (~1.2 it/s). Real: tie with v3 (−0.43, CI −1.06…+0.13). CPU median 1.6 s/message (small: 0.84 s) |
| 2026-10-11 | byt5-base, 20k steps | v3 data, Adafactor, warmup | 20000 | v3 test **1.45**, valid 1.45 | 9.17 | 21.54 | 272 min. Best synthetic by far, but on real two runaway outputs (#103 `уншинаааа…` ×130, #30 emoji line hallucinated and repeated) account for 173 of its 895 edits (the rest ≈ base 7k). Wide CI 6.91–12.53. WER 22.81 (v3: 23.47) |
| 2026-10-12 | byt5-small v4 | `data/v4` (v3 + emoji, stretched letters, hyphenless English suffixes), valid = real_dev | 7000 | v4 test 2.62 | 7.50 | 24.10 | 50 min, best ckpt 5000 (on real_dev). real_dev 5.55 (v3: 5.45, tie). No runaway outputs on real (max pred/ref length 1.28) |
| 2026-10-12 | **byt5-small v4-real** | stage 2 from small v4: real_train ×20 + 10k v4 synthetic (`data/v4_real_mix.jsonl`), lr 1e-4 | 600 | v4 test 2.59 | **6.59** | **30.26** | 7 min, best ckpt 500. real_dev **4.60** (vs small v4 −0.95, CI −1.46…−0.52). Real test vs v3: −1.24, CI −1.76…−0.74 |
| 2026-10-12 | byt5-base v4 | `data/v4`, Adafactor, valid = real_dev | 7000 | v4 test **2.22** | 7.13 | 28.72 | 136 min, best ckpt 7000. real_dev 5.74 (tie with small v4) |
| 2026-10-12 | **byt5-base v4-real** | stage 2 from base v4, same mix, Adafactor, lr 1e-4 | 600 (stopped 400) | v4 test 2.37 | **6.19** | **30.77** | 15 min, best ckpt **100** (early stop). real_dev 4.76: tie with small v4-real (+0.17, CI −0.24…+0.60); real test tie too (−0.40, CI −1.14…+0.24). English kept 76% (small 57%) |
| 2026-10-12 | byt5-base v4-real, lr 3e-5 | as base v4-real, lr 3e-5 | 600 | v4 test 2.27 | 6.51 | 29.23 | 22 min, best ckpt 300. real_dev 4.93: tie with lr 1e-4 (+0.17, CI −0.09…+0.44). Lower lr did not help (`results/v4b`) |
| 2026-10-12 | byt5-small v4-real, 1200 steps | as small v4-real, max 1200 steps | 1200 (stopped 800) | v4 test 2.60 | 6.63 | 29.23 | 8 min, best ckpt 500 again. real_dev 4.53: tie with 600 steps (−0.07, CI −0.26…+0.15) |
| 2026-10-15 | **byt5-small chat** (stage 3) | from small v4-real: `data/chat_mix.jsonl` = 12,978 chat pairs (2,429 Cyrillic chat lines: 238 hand-written + Gemma-expanded, filtered; ×6 Latin styles via `tools/chat_data.py`) + real_train ×20 + 10k v4; valid = real_dev + chat valid | 1000 (stopped 600) | chat valid 1.96 (v4-real: 4.25) | 6.53 (tie, +0.06, CI −0.31…+0.55) | 27.18 | 8 min, best ckpt 300. **Dummy chat (40 msgs, eval-only): 9.76 → 6.58**, EM 32.5 → 55.0, CI of the gap +1.07…+5.34. real_dev 4.47 (tie). Left: ъя endings (yvy → явай), digit+suffix without hyphen (5d → 5D), brand casing (lg, kvm), "min" → минь. `results/chat/` |

**Downstream retrieval (RAG) 2026-10-12** (`eval/retrieval_eval.py`, `results/rag/retrieval.md`). Knowledge base:
Mongolian Wikipedia 20231101 (24,048 articles → 90,709 passages of ~100 words). Each real message is a search query;
the score is how much of the hand-made Cyrillic query's top-10 a query version also finds (overlap@10, %), with
paired-bootstrap 95% CIs. Real test (n=195):

| Query | BM25 overlap@10 | e5 overlap@10 | e5 top-1 agreement |
|---|---:|---:|---:|
| raw Latin | 6.5 | 9.7 | 7.2 |
| naive rules | 20.5 | 40.4 | 29.7 |
| mns5217 rules | 18.3 | 34.2 | 24.1 |
| byt5-small v3 | 70.6 | 77.9 | 74.9 |
| **small v4-real** | **75.0** | **81.4** | 76.4 |
| **base v4-real** | 74.9 | 81.3 | **77.4** |

Raw Latin is nearly unsearchable, even for the multilingual embedding model. v4-real beats v3 on both retrievers
(CIs exclude 0, same on real_dev), so the ~1.2 CER gain matters downstream; small and base tie. Caveat: the gold
query's top-10 is a proxy target (YouTube comments are not information needs); a Latin Q&A set with known answer
articles is the next step.

**LLM comparison 2026-10-12** (`eval/llm_translit.py --provider ollama --model gemma3:12b`, local, temperature 0,
`results/llm/`). Same annotation rules as the system prompt; examples only from real_train. Real test CER
(real_dev after the дүү→дуу fix in brackets):

| System | Real CER ↓ | Real EM ↑ | Eng kept |
|---|---:|---:|---:|
| **byt5-small v4-real (0.3B)** | **6.59** (4.56) | **30.26** | 56.5 |
| Gemma 3 12B, zero-shot | 18.14 (18.95) | 7.18 | 84.8 |
| Gemma 3 12B, 5 random real_train examples | 16.88 (15.59) | 8.21 | 80.4 |
| Gemma 3 12B, 5 retrieved examples (RAG) | 12.49 (12.18) | 16.41 | 71.7 |

Paired bootstrap (real test): small v4-real beats every Gemma condition (vs RAG −5.90, CI −6.94…−4.69). Retrieved
examples beat random ones by 4.39 (CI 3.17…5.75); random vs zero-shot is a tie (CI −0.62…+3.79). One dev zero-shot
reply hit the length limit and counts as empty. Claude/Gemini APIs not run yet (no key).

**Reference corrections 2026-10-10** (annotator's rules, applied to all 195 by `tools/real_test.py edit
data/real_test_corrections.csv`; 11 references changed): шд → шүү дээ, and English words stay in Latin as written
(лайк → like, коммент → comment, клип → clip, Ваав → Wow). Re-scored from the saved predictions
(`eval/bootstrap.py --refs data/real_test.jsonl`); numbers from here on use the corrected references:

| System | Real CER ↓ (95% CI) | Real EM ↑ |
|---|---|---:|
| naive / mns5217 | 19.66 / 23.34 | 1.03 / 0.00 |
| byt5-small v1 | 13.39 (12.18–14.76) | 5.13 |
| byt5-small v2 | 8.15 (7.09–9.25) | 20.00 |
| **byt5-small v3** | **7.83 (6.80–8.90)** | **24.62** |
| byt5-small v3, lr 5e-4 | 8.44 (7.11–9.98) | 23.08 |

v1 → v2/v3 is a real gain (intervals for the difference exclude 0); v2, v3 and lr 5e-4 are tied.

**Bug found 2026-10-10:** transformers 5 removed `warmup_ratio` and `group_by_length`, and `train_byt5.py`
silently dropped unknown settings. So v1–v3 and lr 5e-4 trained **without warmup and without length-grouped
batches**. Fixed (mapped to `warmup_steps` / `train_sampling_strategy`, and the script now warns about dropped
settings). Pretrained runs worked anyway; the from-scratch run did not.

## Training methods to compare

Every method is scored with the same `eval/evaluate.py` run (metric definitions unchanged) and added to the
results log above.

**Scoring**
- **Primary:** real-test CER (195 hand-annotated comments). Secondary: real WER, exact match, ambiguous-letter
  accuracy (ө/о, ү/у, х, и/й/ь/ъ), English words kept, synthetic-test CER.
- **Speed:** ms per message on CPU, because the demo and API run on CPU.
- **Choosing without peeking:** checkpoints and hyperparameters are chosen on synthetic valid CER (or on a separate
  real dev set, see 9). The real test is only reported, never used to pick a winner. Otherwise its score stops
  being trustworthy.
- **Noise:** 195 messages is small. Differences of about 1 CER point may be chance; add bootstrap confidence
  intervals before claiming one method beats another.

| # | Method | What changes | GPU time (est.) | Status | Real CER ↓ | Real EM ↑ | Synth CER ↓ |
|---|---|---|---|---|---:|---:|---:|
| 0 | Rule baselines (naive / mns5217) | greedy Latin→Cyrillic tables | – | done | 19.33 / 23.08 | 1.03 / 0.00 | 17.76 / 19.84 |
| 1 | ByT5-small v1 | Wikipedia 200k pairs, 5k steps | 1.7 h | done | 13.46 | 4.62 | 2.84 |
| 2 | ByT5-small v2 | FineWeb-2 1M pairs + cleanup + chat style, 7k steps | 1.9 h | done | **8.61** | **20.00** | 3.68 |
| 2b | ByT5-small v3 | v2 data + no punctuation in tgt when the writer typed none (`data/v3`) | 1 h | done (tie with v2) | 8.34 | 24.10 | 2.14 |
| 3 | Decoding settings | beam 1 / 4 / 8 (no retraining) | minutes | done: keep beam 4 | 8.88 / 8.34 / 8.30 | 24.10 | 2.18 / 2.15 / 2.14 (valid) |
| 4 | Learning rate sweep | lr 5e-4 on v3 data, 7k steps | 1 h | done (tie on real) | 8.95 | 22.56 | 1.97 (valid) |
| 5 | Longer training | ByT5-base to 20k steps (v3 data, Adafactor) | 4.5 h | done: best synthetic, real hurt by 2 runaway outputs | 9.17 | 21.54 | 1.45 |
| 6 | ByT5-base | 580M instead of 300M parameters; Adafactor, bs 8×4 to fit 12 GB | 1.7 h | done: better synthetic, tie on real; 2× slower on CPU | 7.40 | 23.59 | 1.81 |
| 6b | Controls for 6 | small + warmup fix (AdamW); small + Adafactor | 2 × 1 h | done: both tie with v3 | 7.61 / 7.52 | 22.56 / 24.10 | 2.18 / 2.12 |
| 7 | mT5-small | subword tokenizer instead of bytes (shorter sequences, faster) | ~2 h | todo | | | |
| 8 | ByT5-small from scratch | same architecture, random weights; shows what pretraining adds | 2 × 1 h | done: does not converge in 7k steps (with or without warmup) | 100 | 0 | 100+ (valid) |
| 9 | Two-stage: synthetic → real | fine-tune the best model on 300–500 hand-annotated comments from *new* videos (never the test set) | minutes + annotation time | needs data | | | |
| 10 | LLM baseline (zero/few-shot) | ask a general LLM to convert; strong reference point | – (API cost) | done with Gemma 3 12B (local): zero 18.14 / random-5 16.88 / retrieved-5 12.49 | 12.49 | 16.41 | – |
| 11 | BiLSTM tagger (no pretraining) | byte → output-string labels, 2-layer BiLSTM 8.7M params (`train/train_lstm_tagger.py`) | 10 min | done: int8 ONNX 8.5 MB, 2 ms/message; beats #12 on dev | 8.52 | 23.1 | dev-mix 5.21 |
| 12 | ByT5 encoder tagger (pruned, pretrained) | same labels, 4-block pruned encoder 27.7M (`train/train_tagger.py`) | 1 h | done: int8 ONNX 34 MB, 2 ms/message | 8.77 | 22.6 | dev-mix 5.74 |
