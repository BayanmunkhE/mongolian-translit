"""Evaluation metrics for Latin -> Cyrillic transliteration.

Uses rapidfuzz for fast edit distance / alignment when installed and falls back
to a pure-Python implementation otherwise (identical results, just slower).
"""
from __future__ import annotations

import re
from collections import Counter

try:
    from rapidfuzz.distance import Levenshtein as _RF
except ImportError:  # pragma: no cover
    _RF = None

# Letter groups the task is ambiguous about (keys are report column names).
AMBIGUOUS_GROUPS = {
    "ө/о": set("өо"),
    "ү/у": set("үу"),
    "х": set("х"),
    "и/й/ь/ъ": set("ийьъ"),
    "ш/щ": set("шщ"),
}
AMBIGUOUS_CHARS = set().union(*AMBIGUOUS_GROUPS.values())
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


# ---------------------------------------------------------------------------
# Edit distance + alignment
# ---------------------------------------------------------------------------

def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or token lists)."""
    if _RF is not None:
        return _RF.distance(a, b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def align(ref, hyp) -> list[tuple[int | None, int | None]]:
    """Minimum-edit alignment as (ref_idx, hyp_idx) pairs; None marks ins/del."""
    if _RF is not None:
        pairs, i, j = [], 0, 0
        for op in _RF.editops(ref, hyp):
            while i < op.src_pos and j < op.dest_pos:
                pairs.append((i, j))
                i, j = i + 1, j + 1
            if op.tag == "replace":
                pairs.append((i, j)); i, j = i + 1, j + 1
            elif op.tag == "delete":
                pairs.append((i, None)); i += 1
            else:
                pairs.append((None, j)); j += 1
        while i < len(ref) and j < len(hyp):
            pairs.append((i, j)); i, j = i + 1, j + 1
        pairs += [(k, None) for k in range(i, len(ref))]
        pairs += [(None, k) for k in range(j, len(hyp))]
        return pairs

    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1]


# ---------------------------------------------------------------------------
# Per-example helpers
# ---------------------------------------------------------------------------

def norm(s: str, case_insensitive: bool = False) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower() if case_insensitive else s


def cer(ref: str, hyp: str) -> float:
    ref, hyp = norm(ref), norm(hyp)
    return edit_distance(ref, hyp) / max(1, len(ref))


def ambiguous_counts(ref: str, hyp: str) -> dict[str, list[int]]:
    """{group: [correct, total]} over ref characters in each ambiguous group.

    Compared case-insensitively: this measures letter identity, not casing.
    A ref letter aligned to nothing (deleted) counts as wrong.
    """
    ref, hyp = norm(ref, True), norm(hyp, True)
    counts = {g: [0, 0] for g in AMBIGUOUS_GROUPS}
    for ri, hi in align(ref, hyp):
        if ri is None or ref[ri] not in AMBIGUOUS_CHARS:
            continue
        for g, chars in AMBIGUOUS_GROUPS.items():
            if ref[ri] in chars:
                counts[g][1] += 1
                counts[g][0] += int(hi is not None and hyp[hi] == ref[ri])
    return counts


def latin_words(s: str) -> Counter:
    return Counter(LATIN_WORD_RE.findall(s))


def english_preserved(ref: str, hyp: str) -> tuple[int, int]:
    """(preserved, total) Latin-script words of the reference found verbatim in hyp."""
    r, h = latin_words(ref), latin_words(hyp)
    return sum(min(c, h[w]) for w, c in r.items()), sum(r.values())


# ---------------------------------------------------------------------------
# Corpus-level metrics
# ---------------------------------------------------------------------------

def corpus_metrics(refs: list[str], hyps: list[str]) -> dict[str, float]:
    assert len(refs) == len(hyps)
    m = Counter()
    amb = {g: [0, 0] for g in AMBIGUOUS_GROUPS}
    for ref, hyp in zip(refs, hyps):
        for ci, suffix in ((False, ""), (True, "_ci")):
            r, h = norm(ref, ci), norm(hyp, ci)
            m["char_err" + suffix] += edit_distance(r, h)
            m["chars" + suffix] += len(r)
            rw, hw = r.split(), h.split()
            m["word_err" + suffix] += edit_distance(rw, hw)
            m["words" + suffix] += len(rw)
            m["exact" + suffix] += int(r == h)
        for g, (c, t) in ambiguous_counts(ref, hyp).items():
            amb[g][0] += c
            amb[g][1] += t
        p, t = english_preserved(ref, hyp)
        m["eng_ok"] += p
        m["eng_total"] += t

    n = max(1, len(refs))
    out = {
        "n": len(refs),
        "cer": m["char_err"] / max(1, m["chars"]),
        "wer": m["word_err"] / max(1, m["words"]),
        "exact": m["exact"] / n,
        "cer_ci": m["char_err_ci"] / max(1, m["chars_ci"]),
        "wer_ci": m["word_err_ci"] / max(1, m["words_ci"]),
        "exact_ci": m["exact_ci"] / n,
    }
    tot_c = sum(c for c, _ in amb.values())
    tot_t = sum(t for _, t in amb.values())
    out["ambig_acc"] = tot_c / tot_t if tot_t else float("nan")
    for g, (c, t) in amb.items():
        out[f"acc[{g}]"] = c / t if t else float("nan")
    out["eng_preserved"] = m["eng_ok"] / m["eng_total"] if m["eng_total"] else float("nan")
    out["eng_total"] = m["eng_total"]
    # Added 2026-10-16 (the course rubric lists BLEU): corpus chrF and BLEU via sacrebleu, on the case-sensitive
    # normalised strings. Optional dependency; the existing metrics above are unchanged.
    try:
        import sacrebleu
        norm_refs = [norm(r) for r in refs]
        norm_hyps = [norm(h) for h in hyps]
        out["chrf"] = sacrebleu.corpus_chrf(norm_hyps, [norm_refs]).score / 100
        out["bleu"] = sacrebleu.corpus_bleu(norm_hyps, [norm_refs], tokenize="13a").score / 100
    except ImportError:
        out["chrf"] = out["bleu"] = float("nan")
    return out
