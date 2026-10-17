# Монгол латин гар (prototype)

A pinyin-style Android keyboard: type Mongolian in Latin letters, pick the Cyrillic candidate from the
suggestion strip. Space commits the first candidate; tapping another one commits it and the keyboard remembers
that choice for the word (user dictionary).

How a word is converted, in order:
1. the user's earlier pick for that word (SharedPreferences);
2. `assets/word_dict.json`: the 30k most frequent Latin words with up to 3 teacher candidates (instant);
3. the on-device **tagger** (`assets/tagger.int8.onnx`, BiLSTM 3×512 exported by `tools/export_tagger_onnx.py --lstm`,
   19 MB): one forward pass labels every byte with its output string, run on the word with its two previous
   Latin words as context, ~0.3 ms per word on a desktop core;
4. the seq2seq model (`byt5-pruned-4e2d-1024-ft2`, ONNX int8 encoder 28 MB + decoder 20 MB, ~8 ms per word)
   only when the tagger met a byte it has no label for.

The dictionary candidates are re-ranked by `assets/bigrams.json` (how often each follows the previous committed
word). The raw Latin word is always the last candidate, so English words and names can be kept as typed.

## Build

1. On the PC, pack the assets (they are not in git):
   ```
   python tools/export_onnx.py --model models/byt5-pruned-4e2d-1024-ft2 --out models/onnx/pruned-4e2d-1024-ft2 --check
   python tools/word_dict.py --top 50000
   python tools/pack_keyboard_assets.py
   ```
2. Open `keyboard/android` in Android Studio (Hedgehog or newer, JDK 17), let it sync, run on a device with
   USB debugging (minSdk 26). Without Android Studio: JDK 17 + Gradle 8.9 + the SDK command-line tools
   (`sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"`), `local.properties` with
   `sdk.dir=<sdk>`, then `gradle assembleDebug` → `app/build/outputs/apk/debug/app-debug.apk`; install with
   `adb install -r`, enable with `adb shell ime enable mn.translit.keyboard/.MnLatinIme` and
   `adb shell ime set mn.translit.keyboard/.MnLatinIme`. `tools/start_emulator.ps1` does all of that on the
   Pixel 6 / API 34 emulator (verified 2026-10-18, screenshots in `results/keyboard/`).
3. On the phone: Settings → System → Languages & input → On-screen keyboard → Manage → enable
   "Монгол латин гар", then pick it from the keyboard switcher in any text field.

## Known limits of this prototype
- Words are converted without context, so у/ү/ө choices that need the neighbouring words can be wrong; the
  candidate strip is the mitigation. The server demo converts whole messages with context.
- No long-press, no number row, no autocorrect; the key layout is a plain QWERTY grid.
- Latency and memory are untested on real phones; iOS is not covered (keyboard extensions have a ~50–70 MB
  memory limit, so the model would need to be smaller first).
