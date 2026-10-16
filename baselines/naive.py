"""Baseline 2: "naive informal" rules — one fixed guess per Latin letter.

u -> у, v -> ү, h -> х, o -> о, w -> в, x -> х, c -> ц, ... This is what a
simple keyboard-layout style converter does; it cannot resolve u = у/ү/ө,
o = о/ө, dropped double vowels, or English words.
"""
from __future__ import annotations

from .rules import GreedyRuleTransliterator

NAIVE_TABLE = {
    "kh": "х", "ts": "ц", "ch": "ч", "sh": "ш",
    "ye": "е", "yo": "ё", "yu": "ю", "ya": "я",
    "ö": "ө", "ü": "ү",
    "a": "а", "b": "б", "c": "ц", "d": "д", "e": "э", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "ү", "w": "в", "x": "х", "y": "й", "z": "з",
}


def build() -> GreedyRuleTransliterator:
    return GreedyRuleTransliterator(NAIVE_TABLE, name="naive")


if __name__ == "__main__":
    import sys
    tr = build()
    for line in sys.stdin:
        print(tr(line.rstrip("\n")))
