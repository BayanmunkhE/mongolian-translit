"""Keyboard suggestions for the web "keyboard mode" (app/static/keyboard.html), mirroring the Android IME:

  1. the packed dictionary (data/keyboard/word_dict.json: Latin word -> teacher candidates), re-ranked by how often
     each candidate follows the previous committed Cyrillic word (data/keyboard/bigrams.json);
  2. the ONNX tagger (models/onnx/<name>, tools/export_tagger_onnx.py) run on "prev1 prev2 word", keeping only
     the part after the prefix's own conversion (context-aware, ~2 ms);
  3. an optional seq2seq fallback for words the tagger could not fully label;
  4. the raw Latin word last, so English words and names can be kept as typed.

Dictionary / bigram ranking and casing need no model, so they are unit-tested on their own.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def match_case(latin: str, word: str) -> str:
    """The models capitalise a lone word like a sentence start; mid-sentence the typed casing wins."""
    if latin and latin[0].islower() and word and word[0].isupper():
        if len(word) > 1 and word.isupper():
            return word  # all-caps brands (LG, HD) stay
        return word[0].lower() + word[1:]
    if latin and latin[0].isupper() and word and word[0].islower():
        return word[0].upper() + word[1:]
    return word


def rank_dictionary(cands: list[str], follow: dict[str, int] | None) -> list[str]:
    """Stable sort by bigram count after the previous word; without counts the teacher order stays."""
    if not follow:
        return list(cands)
    return sorted(cands, key=lambda w: -follow.get(w.lower(), 0))


def tail_after_prefix(full: str, prefix_conv: str, n_words: int) -> str:
    """The part of `full` that follows the known conversion of the prefix; falls back to the last n words."""
    if prefix_conv and full.lower().startswith(prefix_conv.lower()):
        rest = full[len(prefix_conv):].strip()
        if rest:
            return rest
    words = full.split()
    return " ".join(words[-n_words:]) if words else full


def merge(*groups: list[str], latin: str) -> list[str]:
    """Best first, case-insensitive dedupe, the typed Latin last."""
    out: list[str] = []
    seen: set[str] = set()
    for g in groups:
        for w in g:
            w = match_case(latin, w.strip())
            if w and w.lower() not in seen:
                seen.add(w.lower())
                out.append(w)
    if latin.lower() not in seen:
        out.append(latin)
    return out


class KeyboardSuggester:
    def __init__(self, tagger_dir: str | Path | None = None, dict_path: str | Path | None = None,
                 bigram_path: str | Path | None = None, seq2seq=None, threads: int = 2):
        self.dict = self._load_dict(dict_path or ROOT / "data" / "keyboard" / "word_dict.json")
        self.bigrams = self._load_json(bigram_path or ROOT / "data" / "keyboard" / "bigrams.json")
        self.tagger = None
        self.tagger_name = None
        if tagger_dir and Path(tagger_dir).exists():
            from translit.onnx_backend import OnnxTaggerTransliterator
            self.tagger = OnnxTaggerTransliterator(str(tagger_dir), threads=threads)
            self.tagger_name = Path(tagger_dir).name
        self.seq2seq = seq2seq  # anything with .nbest(text, n); only used when the tagger left bytes unlabelled

    @staticmethod
    def _load_json(path):
        path = Path(path)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @classmethod
    def _load_dict(cls, path) -> dict[str, list[str]]:
        raw = cls._load_json(path)
        return {w: [c[0] for c in v["candidates"]] for w, v in raw.items()}

    def suggest(self, word: str, context: list[str] | None = None, prev_cyr: str | None = None,
                n: int = 4) -> dict:
        word = word.strip()
        if not word:
            return {"candidates": [], "source": None}
        context = [c for c in (context or []) if c.strip()][-2:]
        follow = self.bigrams.get(prev_cyr.lower()) if prev_cyr else None
        from_dict = rank_dictionary(self.dict.get(word.lower(), []), follow)
        from_model: list[str] = []
        source = "dict" if from_dict else None
        if self.tagger is not None:
            text = " ".join(context + [word])
            full = self.tagger.transliterate(text)
            if context:
                prefix_conv = self.tagger.transliterate(" ".join(context))
                full = tail_after_prefix(full, prefix_conv, 1)
            from_model = [full]
            source = "tagger"
            if "<unk>" in self.tagger.vocab and self.seq2seq is not None and self._has_unknown(text):
                from_model = [t for t, _ in self.seq2seq.nbest(word, 3)] + from_model
                source = "seq2seq"
        cands = merge(from_model, from_dict, latin=word)
        return {"candidates": cands[:n] if n else cands, "source": source}

    def _has_unknown(self, text: str) -> bool:
        """True when the tagger copied a character because its label was <unk>."""
        from translit.onnx_backend import encode
        np = self.tagger.np
        ids = np.array([encode(text)], dtype=np.int64)
        pred = self.tagger.sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
        unk = self.tagger.vocab.index("<unk>")
        return any(int(p) == unk for p in pred[:-1])
