"""Mixed-script messages: keep Cyrillic words as typed and convert only the Latin runs between them.

The model was trained on Latin input only. Given Cyrillic it rewrites it ("Би маргааш өглөө ирнэ" -> "Би мааргааных
шиглээ ирнээ"), but real chats mix both scripts, often in one message. So the app sends only the maximal runs of
non-Cyrillic words that contain a Latin letter to the model; Cyrillic words, and runs of only numbers or emoji, are
copied unchanged. A message without Cyrillic goes to the model whole, exactly as before.
"""
from __future__ import annotations

import re
from typing import Callable

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")
# A run of whole words without any Cyrillic letter, separated by whitespace (a word with one Cyrillic letter,
# e.g. "motive-д", counts as Cyrillic and is kept).
RUN = re.compile(r"(?:^|(?<=\s))[^\sЀ-ӿ]+(?:\s+[^\sЀ-ӿ]+)*(?=\s|$)")


def convert_mixed(text: str, convert: Callable[[str], str]) -> str:
    if not CYRILLIC.search(text):
        return convert(text)
    out, last = [], 0
    for m in RUN.finditer(text):
        span = m.group()
        out.append(text[last:m.start()])
        if LATIN.search(span):
            conv = convert(span)
            # mid-message runs keep the writer's lowercase start (the model may capitalize a "first" word)
            if m.start() > 0 and span[0].islower() and conv[:1].isupper():
                conv = conv[0].lower() + conv[1:]
            out.append(conv)
        else:
            out.append(span)
        last = m.end()
    out.append(text[last:])
    return "".join(out)
