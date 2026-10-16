---
title: Mongolian Latin to Cyrillic
emoji: 🔤
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Informal Latin-script Mongolian → Cyrillic (ByT5) + search
---

# Латин → Кирилл (Mongolian transliteration demo)

Research prototype from an AI Engineer course capstone. A fine-tuned `google/byt5-small` converts informal
Latin-script Mongolian (`za bi margaash ugluu irne`) to Cyrillic (`За би маргааш өглөө ирнэ`), next to a rule-based
MNS 5217:2012 baseline. The search card looks the message up in Mongolian Wikipedia twice, with the raw Latin text
and with its Cyrillic conversion, to show why the conversion matters.

- Model: ByT5-small fine-tuned on ~1M synthetic pairs and hand-corrected YouTube comments; CPU inference.
- Search: `intfloat/multilingual-e5-base` over ~90k passages of Mongolian Wikipedia (CC BY-SA 4.0).
- Messages typed here are not stored by the app.
