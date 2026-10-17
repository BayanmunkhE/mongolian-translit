import json

from translit.keyboard import KeyboardSuggester, match_case, merge, rank_dictionary, tail_after_prefix


def test_match_case_follows_the_typed_word():
    assert match_case("bn", "Байна") == "байна"
    assert match_case("Bn", "байна") == "Байна"
    assert match_case("lg", "LG") == "LG"


def test_rank_dictionary_uses_bigram_counts_and_keeps_order_on_ties():
    assert rank_dictionary(["Мин", "Минь"], {"минь": 10}) == ["Минь", "Мин"]
    assert rank_dictionary(["Мин", "Минь"], None) == ["Мин", "Минь"]
    assert rank_dictionary(["а", "б", "в"], {"б": 1, "в": 1}) == ["б", "в", "а"]


def test_tail_after_prefix():
    assert tail_after_prefix("за би маргааш", "за би", 1) == "маргааш"
    assert tail_after_prefix("за би маргааш", "өөр", 1) == "маргааш"  # prefix mismatch: last word


def test_merge_dedupes_and_puts_latin_last():
    assert merge(["байна"], ["Байна", "байн"], latin="bn") == ["байна", "байн", "bn"]
    assert merge([], [], latin="ok") == ["ok"]
    assert merge(["ok"], [], latin="ok") == ["ok"]


def test_suggester_without_models_uses_dictionary_and_bigrams(tmp_path):
    d = tmp_path / "dict.json"
    d.write_text(json.dumps({"min": {"count": 5, "candidates": [["Мин", 0.6], ["Минь", 0.4]]}}), encoding="utf-8")
    b = tmp_path / "bigrams.json"
    b.write_text(json.dumps({"найз": {"минь": 20}}), encoding="utf-8")
    ks = KeyboardSuggester(tagger_dir=None, dict_path=d, bigram_path=b)
    assert ks.suggest("min")["candidates"] == ["мин", "минь", "min"]
    assert ks.suggest("min", prev_cyr="Найз")["candidates"] == ["минь", "мин", "min"]
    assert ks.suggest("xyz")["candidates"] == ["xyz"]
    assert ks.suggest("  ")["candidates"] == []
