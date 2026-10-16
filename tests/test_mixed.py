"""Mixed-script handling (no torch: a fake converter stands in for the model): pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translit.mixed import convert_mixed  # noqa: E402

calls = []


def fake(text):
    calls.append(text)
    return text.upper()


def test_pure_latin_goes_to_the_model_whole():
    calls.clear()
    assert convert_mixed("za bi margaash irne", fake) == "ZA BI MARGAASH IRNE"
    assert calls == ["za bi margaash irne"]


def test_pure_cyrillic_is_untouched_and_skips_the_model():
    calls.clear()
    assert convert_mixed("Би маргааш өглөө ирнэ", fake) == "Би маргааш өглөө ирнэ"
    assert calls == []


def test_only_latin_runs_are_converted():
    calls.clear()
    text = "Сайн байна уу, margaash hed tsagt uulzah ve"
    assert convert_mixed(text, fake) == "Сайн байна уу, mARGAASH HED TSAGT UULZAH VE"  # mid-message: lowercase start
    assert calls == ["margaash hed tsagt uulzah ve"]


def test_numbers_emoji_and_mixed_words_are_kept():
    calls.clear()
    assert convert_mixed("Улаанбаатар 2024 ❤ онд motive-д", fake) == "Улаанбаатар 2024 ❤ онд motive-д"
    assert calls == []
    assert convert_mixed("за bi\nmargaash ирнэ", fake) == "за bI\nMARGAASH ирнэ"
