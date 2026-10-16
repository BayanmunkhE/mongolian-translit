"""Greedy longest-match Latin -> Cyrillic rule engine shared by the baselines."""
from __future__ import annotations


class GreedyRuleTransliterator:
    """Scans left to right, always applying the longest matching Latin key.

    Case is preserved: "Kh" / "KH" -> "Х", "kh" -> "х". Characters with no rule
    (digits, punctuation, letters outside the table) are copied unchanged.
    """

    def __init__(self, table: dict[str, str], name: str):
        self.name = name
        self.table = {k.lower(): v for k, v in table.items()}
        self.max_len = max(len(k) for k in self.table)

    def transliterate(self, text: str) -> str:
        out, i, low = [], 0, text.lower()
        while i < len(text):
            for n in range(min(self.max_len, len(text) - i), 0, -1):
                key = low[i:i + n]
                if key in self.table:
                    out.append(self._case(text[i:i + n], self.table[key]))
                    i += n
                    break
            else:
                out.append(text[i])
                i += 1
        return "".join(out)

    __call__ = transliterate

    def batch(self, texts: list[str]) -> list[str]:
        return [self.transliterate(t) for t in texts]

    @staticmethod
    def _case(src: str, cyr: str) -> str:
        if not src[0].isupper():
            return cyr
        if len(src) > 1 and src.isupper():
            return cyr.upper()
        return cyr[:1].upper() + cyr[1:]
