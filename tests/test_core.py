"""Fast tests for everything that does not need torch: pytest -q"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_pairs as gp  # noqa: E402
from baselines import get_baseline  # noqa: E402
from eval.metrics import ambiguous_counts, align, cer, corpus_metrics, english_preserved  # noqa: E402


def test_mns_baseline_longest_match_and_case():
    mns = get_baseline("mns5217")
    assert mns("Khüükhed") == "Хүүхэд"
    assert mns("KHAAN") == "ХААН"
    assert mns("tsagaan sar") == "цагаан сар"
    assert mns("yag yum") == "яг юм"
    assert mns("hello") == "hэлло"  # h is not in MNS 5217 -> left as is


def test_naive_baseline():
    naive = get_baseline("naive")
    assert naive("hvvhed") == "хүүхэд"
    assert naive("ugluu") == "углуу"  # cannot know it is өглөө


def test_mns_roundtrip_is_lossless_except_known_merges():
    rng = random.Random(0)
    mns = get_baseline("mns5217")
    style = {"map": gp.MNS, "collapse": 0, "noise": 0}
    s = "Монгол улсын нийслэл Улаанбаатар хот"
    assert mns(gp.romanize(s, style, rng)) == "Монгол улсын нийслэл Улаанбаатар хот".replace("й", "и")


def test_make_pair_and_english_preserved():
    class A:  # minimal args
        p_mns, p_english, p_chat, p_stretch, p_emoji, max_bytes = 0.0, 1.0, 0.0, 0.0, 0.0, 254
    rng = random.Random(1)
    pair = gp.make_pair("Би одоо гэртээ харьж байна.", rng, A)
    eng = pair["style"]["english"]
    assert eng and all(e in pair["src"] and e in pair["tgt"] for e in eng)
    assert not any("Ѐ" <= c <= "ӿ" for c in pair["src"])


def test_legacy_fonts_and_foreign_sentences():
    assert gp.normalize_cyrillic("сvлжээний єєрийн дїїргийн YНДСЭН") == "сүлжээний өөрийн дүүргийн ҮНДСЭН"
    assert gp.normalize_cyrillic("KFC-д орлоо, meeting") == "KFC-д орлоо, meeting"  # pure Latin words untouched
    assert gp.is_foreign("Еліміздің дамуы және экономиканың өсімімен")  # Kazakh
    assert gp.is_foreign("На границах СССР и МНР было сосредоточено много солдат")  # Russian
    assert not gp.is_foreign("захирагч С.Амарсайхан, сайд С.Чинзориг нарт даалгасан байна.")
    assert list(gp.extract_sentences("Энә тухайд тэр хэзээ ч ярьж байсангүй.", 3, 110)) == \
        ["Энэ тухайд тэр хэзээ ч ярьж байсангүй."]


def test_chat_style():
    rng = random.Random(0)
    goy, phrase = gp.abbreviate("Гоё дуу байна уу", 1.0, rng).split(" дуу ")
    assert goy in {"Goy", "Goe", "Goyo"} and phrase in {"bnu", "bnuu", "bn uu"}  # longest phrase wins
    goy, bn = gp.abbreviate("ГОЁ ДУУ БАЙНА", 1.0, rng).split(" ДУУ ")
    assert goy in {"GOY", "GOE", "GOYO"} and bn in {"BN", "BNA", "BNAA"}
    assert gp.abbreviate("байнга", 1.0, rng) == "байнга"  # whole words only
    assert gp.drop_vowels("sonsoj uuliig", 1.0, rng) == "snsj uuliig"

    class A:
        p_mns, p_english, p_chat, p_stretch, p_emoji, max_bytes = 0.0, 0.0, 1.0, 0.0, 0.0, 254
    pairs = [gp.make_pair("Энэ дуу үнэхээр гоё байна.", random.Random(i), A) for i in range(50)]
    assert any("bn" in p["src"].lower() for p in pairs)
    assert all("байна" in p["tgt"].lower() for p in pairs)  # targets stay full Cyrillic


def test_no_punctuation_in_src_means_none_in_tgt():
    class A:
        p_mns, p_english, p_chat, p_stretch, p_emoji, max_bytes = 0.0, 0.3, 0.3, 0.0, 0.0, 254
    pairs = [gp.make_pair("Сайн байна уу, найз аа? Би одоо, гэртээ харьж байна.", random.Random(i), A)
             for i in range(60)]
    stripped = [p for p in pairs if p["style"]["strip_punct"]]
    kept = [p for p in pairs if not p["style"]["strip_punct"]]
    assert stripped and kept
    assert all(not gp.PUNCT_RE.search(p["tgt"]) for p in stripped)
    assert all("?" in p["tgt"] for p in kept)


def test_stretch_and_emoji_keep_reference_clean():
    rng = random.Random(0)
    assert gp.stretch_letters("bi zondoo ih", {"bi", "zondoo"}, rng) == "bi zondoo ih"  # skip set (English)
    stretched = gp.stretch_letters("mash goe", set(), rng)
    assert stretched != "mash goe" and stretched.startswith("mash goe")  # only vowel-final words

    class A:
        p_mns, p_english, p_chat, p_stretch, p_emoji, max_bytes = 0.0, 0.0, 0.0, 1.0, 1.0, 254
    for i in range(40):
        pair = gp.make_pair("Би чамд зөндөө их хайртай шүү.", random.Random(i), A)
        assert pair["style"]["deco"]
        assert re.findall(r"[Ѐ-ӿ]+", pair["tgt"]) == ["Би", "чамд", "зөндөө", "их", "хайртай", "шүү"]
        deco = re.sub(r"[Ѐ-ӿ\s.!?]", "", pair["tgt"])  # the emoji run, copied verbatim
        assert pair["src"].endswith(deco) or pair["src"].startswith(deco)


def test_split_is_deterministic_per_sentence():
    k = gp.sentence_key("За би маргааш ирнэ.")
    assert k == gp.sentence_key("за би маргааш ирнэ")
    assert gp.split_of(k, 0.03, 0.03) == gp.split_of(k, 0.03, 0.03)


def test_bootstrap_matches_corpus_cer():
    from eval.bootstrap import bootstrap, corpus
    stats = {"a": [(1, 10, 0), (0, 5, 1)], "b": [(3, 10, 0), (2, 5, 0)]}
    assert corpus(stats["a"], [0, 1]) == (1 / 15, 0.5)
    summary, cers = bootstrap(stats, ["a", "b"], n=200, seed=0)
    assert summary["a"][0] == 1 / 15 and summary["b"][0] == 5 / 15
    assert all(x <= y for x, y in zip(cers["a"], cers["b"]))  # paired: a never worse here


def test_metrics():
    assert cer("абв", "абв") == 0
    assert abs(cer("абвг", "абг") - 0.25) < 1e-9
    assert align("ab", "b") == [(0, None), (1, 0)]
    c = ambiguous_counts("өглөө", "углуу")
    assert c["ө/о"] == [0, 3]
    assert english_preserved("би meeting-руу орлоо", "би митинг-руу орлоо") == (0, 1)
    m = corpus_metrics(["За би ирнэ"], ["за би ирнэ"])
    assert m["exact"] == 0 and m["exact_ci"] == 1
