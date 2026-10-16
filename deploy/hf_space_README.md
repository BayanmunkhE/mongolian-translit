---
title: Mongolian Latin to Cyrillic
emoji: 🔤
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: gradio_app.py
pinned: false
license: apache-2.0
short_description: Informal Latin-script Mongolian → Cyrillic (ByT5) + search
---

# Латин → Кирилл (Mongolian transliteration demo)

Research prototype from an AI Engineer course capstone. A fine-tuned `google/byt5-small` converts informal
Latin-script Mongolian (`za bi margaash ugluu irne`) to Cyrillic (`За би маргааш өглөө ирнэ`), next to a rule-based
MNS 5217:2012 baseline, and shows the beam alternatives for the words where the model is unsure. The search
buttons look the message up in Mongolian Wikipedia twice, with the raw Latin text and with its Cyrillic
conversion, to show why the conversion matters.

- Model: ByT5-small fine-tuned in three stages (~1M synthetic pairs, hand-corrected YouTube comments, chat data);
  real-test CER 6.2 vs 19.7 for the rule baseline. CPU inference, ~1–2 s per message on the free tier.
- Search: `intfloat/multilingual-e5-base` over ~90k passages of Mongolian Wikipedia (CC BY-SA 4.0).
- Messages typed here are not stored by the app.
