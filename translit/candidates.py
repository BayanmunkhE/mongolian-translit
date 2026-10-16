"""N-best conversions and the word positions where they disagree, for a "pick the right word" UI.

The beam candidates of a message usually differ in one or two words (ө/о, ү/у, a suffix), which is exactly
where the model errs most. Instead of showing three full sentences, the UI shows the best one and a small
choice list on each word where another candidate has a different word.
"""
from __future__ import annotations

import re

from translit.mixed import CYRILLIC, convert_mixed

WS = re.compile(r"\s+")


def candidates_mixed(text: str, model, n: int = 3) -> list[tuple[str, float]]:
    """N-best for a Latin-only message; mixed-script messages get the single mixed conversion."""
    if CYRILLIC.search(text):
        return [(convert_mixed(text, model), 1.0)]
    return model.nbest(text, n)


def word_alternatives(cands: list[str]) -> list[dict]:
    """Positions in the best candidate where another candidate with the same word count has a different word.
    Returns [{"index": i, "options": [other words...]}] in order; candidates that split words differently are skipped
    (they are still shown as whole alternatives)."""
    if not cands:
        return []
    best = WS.split(cands[0].strip())
    options: dict[int, list[str]] = {}
    for other in cands[1:]:
        words = WS.split(other.strip())
        if len(words) != len(best):
            continue
        for i, (a, b) in enumerate(zip(best, words)):
            opts = options.setdefault(i, [])
            if b.lower() != a.lower() and b.lower() not in (o.lower() for o in opts):
                opts.append(b)
    return [{"index": i, "options": opts} for i, opts in sorted(options.items()) if opts]


def pick(text: str, index: int, word: str) -> str:
    """The best candidate with word `index` replaced (what the UI reconstructs client-side)."""
    words = WS.split(text.strip())
    words[index] = word
    return " ".join(words)
