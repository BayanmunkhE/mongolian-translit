#!/usr/bin/env python3
"""Synthetic (informal Latin -> Cyrillic) Mongolian pair generator.

Takes Cyrillic Mongolian text, splits it into sentences, and for every sentence
samples a per-sentence "writing style" (how this writer types ө, ү, х, ц, ж, в,
е, й, ь ...; lowercase or not; dropped double vowels; stripped punctuation;
typos) and romanizes the sentence with it. Optionally inserts English words
(kept verbatim in both src and tgt, sometimes with a Mongolian case suffix:
"meeting-ruu" or "meetingruu" -> "meeting-руу"). Comment-style noise: stretched
letters in src only ("zondoooo" -> "зөндөө") and emoji / "!!!" runs copied to both sides.

Output JSONL, one pair per line:
    {"src": "<latin>", "tgt": "<cyrillic>", "style": {...}}

Train/valid/test are split BY SENTENCE (hash of the normalized sentence), so the
same sentence never appears in two splits, even with different styles.
Sentences that appear in data/real_test.jsonl are excluded from all splits.

Usage:
    python generate_pairs.py --demo
    python generate_pairs.py --hf --max-docs 20000 --out-dir data/
    python generate_pairs.py --input corpus.txt --extra data/seed_conversational.txt --out-dir data/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Letter mappings
# ---------------------------------------------------------------------------

# MNS 5217:2012 (Cyrillic -> Latin). Note: и/й/ъ/ь -> i and ш/щ -> sh are lossy.
MNS = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "ye", "ё": "yo",
    "ж": "j", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "ө": "ö", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ү": "ü", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "sh", "ъ": "i", "ы": "y", "ь": "i", "э": "e", "ю": "yu", "я": "ya",
}

# Informal variants observed in chat/social media: letter -> [(latin, weight)].
INFORMAL = {
    "а": [("a", 1)],
    "б": [("b", 1)],
    "в": [("v", 0.8), ("w", 0.2)],
    "г": [("g", 1)],
    "д": [("d", 1)],
    "е": [("e", 0.55), ("ye", 0.4), ("ie", 0.05)],
    "ё": [("yo", 0.85), ("io", 0.1), ("e", 0.05)],
    "ж": [("j", 0.85), ("zh", 0.15)],
    "з": [("z", 1)],
    "и": [("i", 1)],
    "й": [("i", 0.75), ("y", 0.2), ("j", 0.05)],
    "к": [("k", 1)],
    "л": [("l", 1)],
    "м": [("m", 1)],
    "н": [("n", 1)],
    "о": [("o", 1)],
    "ө": [("u", 0.5), ("o", 0.35), ("ö", 0.15)],
    "п": [("p", 1)],
    "р": [("r", 1)],
    "с": [("s", 1)],
    "т": [("t", 1)],
    "у": [("u", 1)],
    "ү": [("u", 0.45), ("v", 0.4), ("ü", 0.1), ("y", 0.05)],
    "ф": [("f", 1)],
    "х": [("h", 0.65), ("kh", 0.2), ("x", 0.15)],
    "ц": [("ts", 0.6), ("c", 0.35), ("tz", 0.05)],
    "ч": [("ch", 0.85), ("c", 0.15)],
    "ш": [("sh", 1)],
    "щ": [("sh", 0.8), ("sch", 0.2)],
    "ъ": [("i", 0.5), ("", 0.4), ("'", 0.1)],
    "ы": [("ii", 0.5), ("y", 0.35), ("i", 0.15)],
    "ь": [("i", 0.5), ("", 0.4), ("'", 0.1)],
    "э": [("e", 1)],
    "ю": [("yu", 0.8), ("iu", 0.2)],
    "я": [("ya", 0.85), ("ia", 0.15)],
}
assert set(MNS) == set(INFORMAL)

# Letters whose chosen variant is worth recording in the style dict.
STYLE_KEYS = [c for c, opts in INFORMAL.items() if len(opts) > 1]

# Long vowels / diphthongs that informal writers often shorten (маргааш -> margash).
COLLAPSIBLE = {"аа", "ээ", "ии", "оо", "өө", "уу", "үү", "ий", "ый"}

# Look-alike characters that show up in scraped Mongolian text.
CHAR_FIXES = {
    "ӧ": "ө", "Ӧ": "Ө", "ɵ": "ө", "Ɵ": "Ө", "ӫ": "ө", "Ӫ": "Ө", "є": "ө", "Є": "Ө",
    "ӱ": "ү", "Ӱ": "Ү", "ұ": "ү", "Ұ": "Ү", "ї": "ү", "Ї": "Ү",
    " ": " ", "​": "", "­": "", "ʼ": "'", "’": "'",
}

# Legacy fonts typed some Cyrillic letters with Latin look-alikes (сvлжээ = сүлжээ, YНДСЭН = ҮНДСЭН).
# Only applied inside words that mix the two scripts.
LATIN_LOOKALIKES = str.maketrans("vVYyaceopxACEOPXHKMTB", "үҮҮуасеорхАСЕОРХНКМТВ")

# Letters of Kazakh, Buryat, Kalmyk etc. that Khalkha Mongolian does not use: such sentences are dropped.
FOREIGN_LETTERS = set("ғқңіһҳҷӣӯҡҫӑӗјҙӹҒҚҢІҺҲҶӢӮҠҪӐӖЈҘӸ")
# Lowercase Russian function words; a sentence with >= 2 of them, none of MN_WORDS and no ө/ү is Russian.
RU_WORDS = set("""что это и в на не с по для как от из к о мы вы она они его её ее но же бы ли все всё так
только уже или при если был была были будет есть нет году также который которые которая этого этой этом
может можно очень когда где чтобы даже после через более между под над было много будут этот эти того
тоже здесь там сейчас себя себе нас вас них ему""".split())
MN_WORDS = set("нь бол энэ тэр байна юм болон гэж гэсэн байгаа байсан ба бөгөөд байдаг хийх улс".split())
# ә / җ in otherwise Khalkha text are typos for э / ж ("Энә тухайд").
ODD_LETTER_FIXES = str.maketrans("әӘҗҖ", "эЭжЖ")

PUNCT_RE = re.compile(r"[.,!?;:\"«»“”„()\[\]…—–]|(?<!\w)-(?!\w)")
LATIN_LETTERS = "abcdefghijklmnopqrstuvwxyz"
KEYBOARD_NEIGHBORS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh",
    "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm",
    "o": "ip", "p": "o", "r": "et", "s": "ad", "t": "ry", "u": "yi", "v": "cb",
    "w": "qe", "x": "zc", "y": "tu", "z": "x",
}

# English words people mix into Mongolian chats.
ENGLISH_WORDS = """
meeting deadline email project file link zoom online offline ok okay sorry thanks
laptop phone update team office report presentation slide code bug server app
website password account chat video call game party weekend coffee lunch break
shopping sale order delivery ticket flight hotel gym workout class exam homework
assignment course budget invoice feedback review design data dashboard deploy
download upload screenshot story post like share comment follow live stream
playlist song movie series netflix youtube facebook instagram google excel word
powerpoint wifi password internet charger battery bluetooth headphone mouse
keyboard printer manager boss intern interview salary bonus contract client
customer market startup business plan target task schedule calendar event
training workshop seminar campus dorm lab research paper thesis grade cafe
burger pizza taxi uber bus metro parking traffic
""".split()

# Mongolian case suffixes that get hyphen-attached to English words.
SUFFIXES = ["руу", "рүү", "д", "ыг", "ийг", "аас", "ээс", "тай", "тэй",
            "аа", "ээ", "ийн", "ын", "гаа", "гээ", "ыхаа", "ууд", "үүд", "оо", "өө", "энд", "анд"]
# Marks a segment that is in tgt only: the hyphen of "meeting-руу" when the writer typed "meetingruu".
TGT_ONLY = "tgt"

# Emoji and punctuation runs that commenters add; copied verbatim into the reference.
EMOJI = "😂 😊 😍 ❤ ❤️ 😘 😢 😭 👍 👏 🔥 🙏 💕 🥰 😁 😆 🤣 💪 🇲🇳 ✌ 😅 💯 😔 🥺".split()
DECO_PUNCT = ["!!!", "!!", "...", "..", "??", "?!"]
STRETCH_RE = re.compile(r"[A-Za-z]+[aeiouyvAEIOUYV](?![A-Za-z])")  # words ending in a vowel

# Chat shorthand: Cyrillic word or phrase -> Latin spellings typed in chats and comments.
# Used only by "chat" styles; each match is abbreviated with probability style["abbrev"].
CHAT_ABBREV = {
    "байна": [("bn", 0.6), ("bna", 0.3), ("bnaa", 0.1)],
    "байна уу": [("bnu", 0.5), ("bnuu", 0.3), ("bn uu", 0.2)],
    "байгаа": [("bga", 0.7), ("bgaa", 0.3)],
    "байгаад": [("bgaad", 1)],
    "байх": [("bh", 1)],
    "байсан": [("bsn", 1)],
    "байдаг": [("bdg", 0.6), ("bdag", 0.4)],
    "байж": [("bj", 1)],
    "байхгүй": [("bhgui", 0.6), ("bhgvi", 0.4)],
    "шүү дээ": [("shd", 0.4), ("sht", 0.3), ("shuu de", 0.2), ("shu dee", 0.1)],
    "биз дээ": [("bzde", 0.5), ("biz de", 0.5)],
    "юм": [("ym", 0.7), ("yum", 0.3)],
    "юмаа": [("ymaa", 1)],
    "юм бэ": [("ymbe", 0.5), ("yum be", 0.5)],
    "гэж": [("gj", 0.7), ("gej", 0.3)],
    "гэсэн": [("gsn", 0.7), ("gesn", 0.3)],
    "гоё": [("goy", 0.6), ("goe", 0.3), ("goyo", 0.1)],
    "зүгээр": [("zgr", 0.5), ("zugeer", 0.5)],
    "баярлалаа": [("bayrlalaa", 0.6), ("bayrla", 0.4)],
    "үгүй": [("ugui", 0.5), ("ugv", 0.3), ("ugy", 0.2)],
    "тийм": [("tim", 0.6), ("tiim", 0.4)],
    "хаха": [("hha", 0.3), ("haha", 0.3), ("hh", 0.25), ("hah", 0.15)],
    "хэхэ": [("hehe", 0.6), ("hhe", 0.4)],
    "одоо": [("odo", 1)],
    "чинь": [("chn", 0.5), ("chin", 0.5)],
    "нь": [("n", 1)],
    "бол": [("bl", 1)],
}
CHAT_RE = re.compile(r"(?<![Ѐ-ӿ])(" + "|".join(sorted(map(re.escape, CHAT_ABBREV), key=len, reverse=True))
                     + r")(?![Ѐ-ӿ])", re.IGNORECASE)

DEMO_SENTENCES = [
    "За би маргааш өглөө ирнэ.",
    "Сайн байна уу? Өнөөдөр их завгүй байлаа.",
    "Хүүхдүүд сургуульдаа явсан, би одоо гэртээ харьж байна.",
    "Монгол Улс бол Төв Азид орших далайд гарцгүй улс юм.",
    "Өвөө эмээ дээрээ очиж, хөдөө хэдэн хоног амарна.",
    "Цагаан сар ирэх долоо хоногт болно.",
    "Энэ асуудлыг хурдан шийдэх хэрэгтэй.",
    "Шинэ жилийн мэнд хүргэе!",
]

# ---------------------------------------------------------------------------
# Text cleaning / sentence extraction
# ---------------------------------------------------------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
ALLOWED_RE = re.compile(r"^[Ѐ-ӿA-Za-z0-9\s.,!?;:\-–—'\"«»“”()%/№+&]+$")
CYR_RE = re.compile(r"[Ѐ-ӿ]")
LAT_RE = re.compile(r"[A-Za-z]")


MIXED_WORD_RE = re.compile(r"\w*[Ѐ-ӿ]\w*")


def fix_mixed_words(text: str) -> str:
    """Map Latin look-alikes back to Cyrillic inside words that mix both scripts."""
    return MIXED_WORD_RE.sub(
        lambda m: m.group().translate(LATIN_LOOKALIKES) if LAT_RE.search(m.group()) else m.group(), text)


def normalize_cyrillic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for bad, good in CHAR_FIXES.items():
        text = text.replace(bad, good)
    return fix_mixed_words(re.sub(r"\s+", " ", text).strip())


def is_foreign(s: str) -> bool:
    """Kazakh / Buryat / Kalmyk (letters Khalkha does not use) or Russian sentences."""
    if FOREIGN_LETTERS & set(s):
        return True
    if re.search("[өүӨҮ]", s):
        return False
    words = re.findall(r"[а-яё]+", s)  # lowercase only, so initials like "С." are not prepositions
    return sum(w in RU_WORDS for w in words) >= 2 and not any(w in MN_WORDS for w in words)


def sentence_key(s: str) -> str:
    """Key used for dedup, split assignment and real-test exclusion."""
    s = PUNCT_RE.sub(" ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def is_good_sentence(s: str, min_words: int) -> bool:
    if len(s.split()) < min_words or not ALLOWED_RE.match(s):
        return False
    n_cyr = len(CYR_RE.findall(s))
    n_lat = len(LAT_RE.findall(s))
    return n_cyr >= 8 and n_cyr / max(1, n_cyr + n_lat) >= 0.85


def chunk_words(s: str, max_chars: int) -> list[str]:
    """Split overly long sentences into word chunks of <= max_chars."""
    if len(s) <= max_chars:
        return [s]
    chunks, cur = [], ""
    for w in s.split():
        if cur and len(cur) + 1 + len(w) > max_chars:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) <= max_chars]


def extract_sentences(text: str, min_words: int, max_chars: int):
    for line in text.splitlines():
        line = normalize_cyrillic(line)
        if not line:
            continue
        for s in SENT_SPLIT_RE.split(line):
            for c in chunk_words(s.strip(), max_chars):
                if is_good_sentence(c, min_words) and not is_foreign(c):
                    yield c.translate(ODD_LETTER_FIXES)


# ---------------------------------------------------------------------------
# Style sampling + romanization
# ---------------------------------------------------------------------------

def weighted_choice(rng: random.Random, options):
    vals, weights = zip(*options)
    return rng.choices(vals, weights=weights, k=1)[0]


def sample_style(rng: random.Random, p_mns: float, p_chat: float = 0.0) -> dict:
    preset = "mns" if rng.random() < p_mns else "informal"
    if preset == "mns":
        mapping = dict(MNS)
    else:
        mapping = {c: weighted_choice(rng, opts) for c, opts in INFORMAL.items()}
    chat = preset == "informal" and rng.random() < p_chat
    case = rng.random()
    return {
        "preset": preset,
        "map": mapping,
        "lower": case < 0.72,
        "upper": 0.72 <= case < 0.75,  # ALL CAPS writer: src and tgt both upper case
        "collapse": 0.0 if rng.random() < 0.5 else round(rng.uniform(0.2, 0.9), 2),
        # a writer who types no punctuation gets a reference without it too (as in real_test.jsonl),
        # so the model does not learn to invent commas and full stops
        "strip_punct": rng.random() < 0.6,
        "typo": 0.0 if rng.random() < 0.5 else round(rng.uniform(0.003, 0.03), 3),
        # per-letter chance of deviating from the writer's own convention
        "noise": 0.0 if preset == "mns" else round(rng.uniform(0.0, 0.08), 3),
        # chat shorthand (CHAT_ABBREV) and dropped short vowels (sonsoj -> snsj)
        "abbrev": round(rng.uniform(0.4, 1.0), 2) if chat else 0.0,
        "devowel": round(rng.uniform(0.05, 0.4), 2) if chat and rng.random() < 0.5 else 0.0,
    }


def apply_case(latin: str, upper: bool, word_upper: bool) -> str:
    if not upper or not latin:
        return latin
    return latin.upper() if word_upper else latin[0].upper() + latin[1:]


def romanize(text: str, style: dict, rng: random.Random) -> str:
    mapping, out, i = style["map"], [], 0
    while i < len(text):
        c = text[i]
        lc = c.lower()
        if lc not in mapping:
            out.append(c)
            i += 1
            continue
        step = 1
        if style["collapse"] and text[i:i + 2].lower() in COLLAPSIBLE \
                and rng.random() < style["collapse"]:
            step = 2  # write the long vowel as a single letter
        if style["noise"] and rng.random() < style["noise"]:
            lat = weighted_choice(rng, INFORMAL[lc])
        else:
            lat = mapping[lc]
        nxt = text[i + step] if i + step < len(text) else ""
        prv = text[i - 1] if i > 0 else ""
        word_upper = c.isupper() and (nxt.isupper() or prv.isupper())
        out.append(apply_case(lat, c.isupper(), word_upper))
        i += step
    return "".join(out)


def add_typos(text: str, rate: float, rng: random.Random) -> str:
    if rate <= 0:
        return text
    chars, out, i = list(text), [], 0
    while i < len(chars):
        c = chars[i]
        if c.lower() in LATIN_LETTERS and rng.random() < rate:
            op = rng.choice(["delete", "dup", "swap", "neighbor"])
            if op == "delete":
                i += 1
                continue
            if op == "dup":
                out += [c, c]
            elif op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                out += [chars[i + 1], c]
                i += 1
            else:
                nb = KEYBOARD_NEIGHBORS.get(c.lower(), c.lower())
                out.append(rng.choice(nb))
        else:
            out.append(c)
        i += 1
    return "".join(out)


def abbreviate(text: str, rate: float, rng: random.Random) -> str:
    """Replace chat-shorthand words (CHAT_ABBREV) by their Latin spelling; romanize() keeps them."""
    if rate <= 0:
        return text

    def sub(m):
        word = m.group()
        if rng.random() >= rate:
            return word
        lat = weighted_choice(rng, CHAT_ABBREV[word.lower()])
        if len(word) > 1 and word.isupper():
            return lat.upper()
        return lat.capitalize() if word[0].isupper() else lat

    return CHAT_RE.sub(sub, text)


def drop_vowels(text: str, rate: float, rng: random.Random) -> str:
    """Chat-style vowel dropping (sonsoj -> snsj); keeps word-initial and doubled vowels."""
    if rate <= 0:
        return text
    out = []
    for i, c in enumerate(text):
        prev, nxt = text[i - 1] if i else " ", text[i + 1] if i + 1 < len(text) else " "
        lc = c.lower()
        if (lc in "aeiou" and prev.isalpha() and lc != prev.lower() and lc != nxt.lower()
                and rng.random() < rate):
            continue
        out.append(c)
    return "".join(out)


def stretch_letters(text: str, skip: set[str], rng: random.Random) -> str:
    """Stretch the last letter of 1-2 words (zondoo -> zondoooooo); the reference keeps the normal spelling."""
    words = [m for m in STRETCH_RE.finditer(text) if m.group().lower() not in skip]
    if not words:
        return text
    picks = sorted(rng.sample(words, min(len(words), rng.choice([1, 1, 2]))), key=lambda m: m.start())
    out, pos = [], 0
    for m in picks:
        out += [text[pos:m.end()], m.group()[-1] * rng.randint(2, 8)]
        pos = m.end()
    return "".join(out) + text[pos:]


def decoration(rng: random.Random, punct_ok: bool) -> str:
    """An emoji run (😂😂😂) or, for writers who type no other punctuation, a run like '!!!'."""
    if punct_ok and rng.random() < 0.3:
        return rng.choice(DECO_PUNCT)
    deco = rng.choice(EMOJI) * rng.choice([1, 1, 1, 2, 3, 3, 5])
    if rng.random() < 0.25:
        deco += rng.choice(EMOJI) * rng.choice([1, 2, 3])
    return deco


def insert_english(words: list[str], rng: random.Random, lower: bool):
    """Return segments [(text, transliterate?)] with 1-2 English words inserted."""
    segs = [[(w, True)] for w in words]
    inserted = []
    for _ in range(rng.choice([1, 1, 2])):
        eng = rng.choice(ENGLISH_WORDS)
        if not lower and rng.random() < 0.3:
            eng = eng.capitalize()
        token = [(eng, False)]
        if rng.random() < 0.4:
            # the reference always has the hyphen; some writers omit it (commentoo -> comment-оо)
            hyphen = ("-", False) if rng.random() < 0.6 else ("-", TGT_ONLY)
            token += [hyphen, (rng.choice(SUFFIXES), True)]
        # never position 0: keeps the sentence-initial capital on a Mongolian word
        segs.insert(rng.randint(1, max(1, len(segs))), token)
        inserted.append(eng)
    flat = []
    for k, seg in enumerate(segs):
        if k:
            flat.append((" ", False))
        flat.extend(seg)
    return flat, inserted


def make_pair(sentence: str, rng: random.Random, args) -> dict | None:
    style = sample_style(rng, args.p_mns, args.p_chat)
    style["stretch"] = style["preset"] == "informal" and rng.random() < args.p_stretch
    style["deco"] = rng.random() < args.p_emoji
    if style["upper"]:
        sentence = sentence.upper()
    english = []
    if rng.random() < args.p_english:
        segments, english = insert_english(sentence.split(), rng, style["lower"])
    else:
        segments = [(sentence, True)]

    tgt = "".join(t for t, _ in segments)
    src_parts = []
    for text, translit in segments:
        if translit == TGT_ONLY:
            continue
        if translit:
            latin = romanize(abbreviate(text, style["abbrev"], rng), style, rng)
            latin = drop_vowels(latin, style["devowel"], rng)
            src_parts.append(add_typos(latin, style["typo"], rng))
        else:
            src_parts.append(text)
    src = "".join(src_parts)
    if style["stretch"]:
        src = stretch_letters(src, {e.lower() for e in english}, rng)
    if style["lower"]:
        src = src.lower()
        # English words are copied verbatim, so tgt must match the src casing.
        if english:
            tgt = "".join(t.lower() if not tr else t for t, tr in segments)
    if style["strip_punct"]:
        src = PUNCT_RE.sub(" ", src)
        tgt = re.sub(r"\s+", " ", PUNCT_RE.sub(" ", tgt)).strip()
    src = re.sub(r"\s+", " ", src).strip()
    if style["deco"] and src:
        deco = decoration(rng, style["strip_punct"])
        if rng.random() < 0.15:
            src, tgt = f"{deco} {src}", f"{deco} {tgt}"
        else:
            sep = "" if rng.random() < 0.4 else " "
            src, tgt = src + sep + deco, tgt + sep + deco

    if not src or len(src.encode()) > args.max_bytes or len(tgt.encode()) > args.max_bytes:
        return None
    record_style = {
        "preset": style["preset"],
        "map": {k: style["map"][k] for k in STYLE_KEYS},
        "lower": style["lower"],
        "upper": style["upper"],
        "collapse": style["collapse"],
        "strip_punct": style["strip_punct"],
        "typo": style["typo"],
        "noise": style["noise"],
        "abbrev": style["abbrev"],
        "devowel": style["devowel"],
        "stretch": style["stretch"],
        "deco": style["deco"],
        "english": english,
    }
    return {"src": src, "tgt": tgt, "style": record_style}


# ---------------------------------------------------------------------------
# Corpus loading / splitting
# ---------------------------------------------------------------------------

def split_of(key: str, valid_frac: float, test_frac: float) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 1_000_000 / 1_000_000
    if h < valid_frac:
        return "valid"
    if h < valid_frac + test_frac:
        return "test"
    return "train"


def iter_hf_texts(args):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets  (needed for --hf)")
    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split, streaming=True)
    for i, row in enumerate(ds):
        if args.max_docs and i >= args.max_docs:
            break
        yield row[args.text_field]


def iter_file_texts(paths):
    for p in paths:
        with open(p, encoding="utf-8-sig") as f:  # -sig: tolerate Windows BOM
            for line in f:
                yield line


def load_excluded(paths) -> set[str]:
    keys = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    keys.add(sentence_key(normalize_cyrillic(json.loads(line)["tgt"])))
    return keys


def collect_sentences(texts, args, excluded, seen, stats, source):
    out = []
    # short chat lines ("Ойлголоо, тэгье.") are fine in the hand-written extra corpora
    min_words = min(args.min_words, 2) if source == "extra" else args.min_words
    for text in texts:
        for s in extract_sentences(text, min_words, args.max_chars):
            key = sentence_key(s)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            if key in excluded:
                stats["excluded_real_test"] += 1
                continue
            out.append((s, key, source))
            if args.max_sentences and len(out) >= args.max_sentences:
                return out
    return out


def run_demo(args):
    rng = random.Random(args.seed)
    for s in DEMO_SENTENCES:
        for _ in range(2):
            pair = make_pair(s, rng, args)
            if pair:
                print(f"{pair['src']}\n  -> {pair['tgt']}\n  style: "
                      f"preset={pair['style']['preset']} ө={pair['style']['map']['ө']} "
                      f"ү={pair['style']['map']['ү']} х={pair['style']['map']['х']} "
                      f"collapse={pair['style']['collapse']} abbrev={pair['style']['abbrev']} "
                      f"devowel={pair['style']['devowel']} stretch={pair['style']['stretch']} "
                      f"english={pair['style']['english']}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("sources")
    src.add_argument("--demo", action="store_true", help="print a few examples and exit")
    src.add_argument("--hf", action="store_true", help="stream a HF dataset (default: mn Wikipedia)")
    src.add_argument("--hf-dataset", default="wikimedia/wikipedia")
    src.add_argument("--hf-config", default="20231101.mn")
    src.add_argument("--hf-split", default="train")
    src.add_argument("--text-field", default="text")
    src.add_argument("--max-docs", type=int, default=0)
    src.add_argument("--input", nargs="*", default=[], help="plain-text Cyrillic corpus files")
    src.add_argument("--extra", nargs="*", default=[],
                     help="small conversational corpora, oversampled with --extra-variants")
    src.add_argument("--extra-variants", type=int, default=20)

    gen = ap.add_argument_group("generation")
    gen.add_argument("--out-dir", default="data")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--variants", type=int, default=2, help="styles per train sentence")
    gen.add_argument("--max-sentences", type=int, default=0, help="cap per source (0 = no cap)")
    gen.add_argument("--max-train-pairs", type=int, default=0)
    gen.add_argument("--max-valid", type=int, default=2000, help="cap on valid pairs")
    gen.add_argument("--max-test", type=int, default=2000, help="cap on test pairs")
    gen.add_argument("--valid-frac", type=float, default=0.03)
    gen.add_argument("--test-frac", type=float, default=0.03)
    gen.add_argument("--min-words", type=int, default=3)
    gen.add_argument("--max-chars", type=int, default=110, help="longer sentences are chunked")
    gen.add_argument("--max-bytes", type=int, default=254, help="UTF-8 byte cap for src and tgt")
    gen.add_argument("--p-english", type=float, default=0.25)
    gen.add_argument("--p-mns", type=float, default=0.1, help="share of strict MNS 5217 styles")
    gen.add_argument("--p-chat", type=float, default=0.35,
                     help="share of informal styles that use chat shorthand (bn, bga, goy) and drop vowels")
    gen.add_argument("--p-stretch", type=float, default=0.15,
                     help="share of informal styles that stretch letters (zondoooo); tgt keeps the normal spelling")
    gen.add_argument("--p-emoji", type=float, default=0.25,
                     help="share of pairs with an emoji or '!!!' run, copied verbatim into tgt")
    gen.add_argument("--exclude", nargs="*", default=["data/real_test.jsonl"],
                     help="JSONL files whose tgt sentences must never be generated")
    args = ap.parse_args(argv)

    if args.demo:
        run_demo(args)
        return
    if not (args.hf or args.input or args.extra):
        ap.error("give --hf, --input or --extra (or --demo)")

    excluded = load_excluded(args.exclude)
    stats, seen, sentences = Counter(), set(), []
    if args.hf:
        sentences += collect_sentences(iter_hf_texts(args), args, excluded, seen, stats, "hf")
    if args.input:
        sentences += collect_sentences(iter_file_texts(args.input), args, excluded, seen, stats, "input")
    if args.extra:
        sentences += collect_sentences(iter_file_texts(args.extra), args, excluded, seen, stats, "extra")
    print(f"{len(sentences)} unique sentences ({stats['dup']} duplicates, "
          f"{stats['excluded_real_test']} excluded as real-test)", file=sys.stderr)

    splits = {"train": [], "valid": [], "test": []}
    for s, key, source in sentences:
        split = split_of(key, args.valid_frac, args.test_frac)
        n = args.variants if split == "train" else 1
        if source == "extra":
            n = args.extra_variants if split == "train" else min(3, args.extra_variants)
        for v in range(n):
            # Per-sentence RNG: output is independent of corpus order.
            rng = random.Random(f"{args.seed}|{split}|{v}|{s}")
            pair = make_pair(s, rng, args)
            if pair:
                splits[split].append(pair)
            else:
                stats["too_long"] += 1

    shuffle_rng = random.Random(args.seed)
    caps = {"train": args.max_train_pairs, "valid": args.max_valid, "test": args.max_test}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs in splits.items():
        shuffle_rng.shuffle(pairs)
        if caps[name]:
            pairs = pairs[:caps[name]]
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        stats[f"{name}_pairs"] = len(pairs)
        print(f"{name}: {len(pairs)} pairs -> {out_dir / (name + '.jsonl')}", file=sys.stderr)
    (out_dir / "generation_stats.json").write_text(
        json.dumps({"args": vars(args), "stats": stats}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
