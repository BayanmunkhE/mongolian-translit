from translit.candidates import candidates_mixed, pick, word_alternatives


class FakeModel:
    def __call__(self, text):
        return "конв:" + text

    def nbest(self, text, n=3):
        return [("За би маргааш өглөө ирнэ", 0.9), ("За би маргааш углуу ирнэ", 0.07), ("За би маргааш өглөө ирне", 0.03)]


def test_word_alternatives_only_where_candidates_differ():
    alts = word_alternatives(["За би маргааш өглөө ирнэ", "За би маргааш углуу ирнэ", "За би маргааш өглөө ирне"])
    assert alts == [{"index": 3, "options": ["углуу"]}, {"index": 4, "options": ["ирне"]}]


def test_word_alternatives_skips_candidates_with_other_word_count_and_dedupes():
    alts = word_alternatives(["а б в", "а б", "а х в", "а х в"])
    assert alts == [{"index": 1, "options": ["х"]}]
    assert word_alternatives(["а б"]) == []
    assert word_alternatives(["а Б в", "а б в", "А б В"]) == []  # casing-only differences are not alternatives
    assert word_alternatives([]) == []


def test_pick_replaces_one_word():
    assert pick("За би маргааш өглөө ирнэ", 3, "углуу") == "За би маргааш углуу ирнэ"


def test_candidates_mixed_uses_nbest_only_for_latin_text():
    m = FakeModel()
    assert candidates_mixed("za bi margaash ugluu irne", m)[0][0] == "За би маргааш өглөө ирнэ"
    assert candidates_mixed("ene kino үнэхээр goy bna", m) == [("конв:ene kino үнэхээр конв:goy bna", 1.0)]
