"""Annotation DB tests (stdlib only, no FastAPI needed): pytest -q"""
import csv
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotate import db  # noqa: E402
from tools.youtube_comments import clean_comment, is_latin_mongolian  # noqa: E402


def test_add_save_export(tmp_path):
    con = db.connect(tmp_path / "a.db")
    a = db.add(con, "  za bi  margaash irne ", "messenger")
    assert db.add(con, "ZA BI MARGAASH IRNE") is None  # duplicate (case-insensitive)
    with pytest.raises(ValueError):
        db.add(con, "за bi irne")  # Cyrillic in src
    b = db.add(con, "goy duu bna")

    assert db.next_todo(con)["id"] == a
    with pytest.raises(ValueError):
        db.save(con, a, "", "done")  # empty reference
    item, warnings = db.save(con, a, "За би маргааш ирнэ", "done")
    assert item["status"] == "done" and item["src"] == "za bi margaash irne" and not warnings
    db.save(con, b, "", "skip")
    assert db.next_todo(con) is None
    assert db.stats(con) == {"todo": 0, "done": 1, "skip": 1, "total": 2}
    assert db.neighbor(con, b, -1)["id"] == a and db.neighbor(con, b, 1) is None

    buf = io.StringIO()
    assert db.write_csv(con, buf) == 1
    rows = list(csv.DictReader(io.StringIO(buf.getvalue())))
    assert rows == [{"src": "za bi margaash irne", "tgt": "За би маргааш ирнэ",
                     "source": "messenger", "note": ""}]


def test_youtube_filter():
    assert is_latin_mongolian("2023 ond sonsson hun bnu")
    assert not is_latin_mongolian("this song is so good")
    assert not is_latin_mongolian("Гоё дуу байна шүү")
    assert clean_comment("@user123 bas  goy bn") == "bas goy bn"
