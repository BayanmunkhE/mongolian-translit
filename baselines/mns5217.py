"""Baseline 1: reverse of MNS 5217:2012 (cf. Interscript masm-mon-Latn-Cyrl-5217-2012).

The standard is lossy in the Cyrillic -> Latin direction (и/й/ъ/ь -> i,
ш/щ -> sh), so the reverse picks the most frequent letter: i -> и, sh -> ш.
Letters the standard never produces (h, w, x, c, q) are left untouched, which
is exactly how a strict standard-based converter behaves on informal text.
"""
from __future__ import annotations

from .rules import GreedyRuleTransliterator

MNS5217_REVERSE = {
    # digraphs first (greedy longest match makes order irrelevant, kept for reading)
    "kh": "х", "ts": "ц", "ch": "ч", "sh": "ш",
    "ye": "е", "yo": "ё", "yu": "ю", "ya": "я",
    "ö": "ө", "ü": "ү",
    "a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "j": "ж", "z": "з",
    "i": "и", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "r": "р", "s": "с", "t": "т", "u": "у", "f": "ф", "y": "ы", "e": "э",
}


def build() -> GreedyRuleTransliterator:
    return GreedyRuleTransliterator(MNS5217_REVERSE, name="mns5217")


if __name__ == "__main__":
    import sys
    tr = build()
    for line in sys.stdin:
        print(tr(line.rstrip("\n")))
