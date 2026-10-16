"""Fast tests for the RAG pieces that need neither torch nor an index: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.llm_translit import ExampleRetriever, user_prompt  # noqa: E402
from eval.retrieval_eval import score  # noqa: E402
from rag.corpus import chunk  # noqa: E402


def test_chunk_packs_paragraphs_and_splits_long_ones():
    text = "\n".join(["нэг хоёр гурав."] * 50)  # 50 paragraphs of 3 words
    passages = chunk(text, target=10)
    assert all(len(p.split()) <= 12 for p in passages)
    assert sum(len(p.split()) for p in passages) == 150
    long_para = " ".join(["үг."] * 200)  # one paragraph over max_words -> split at sentence ends
    assert len(chunk(long_para, target=100, max_words=160)) == 2


def test_score_overlap_top1_and_empty_gold():
    results = {"gold": [[(1, 2.0), (2, 1.0)], []],  # second query has no gold results -> skipped
               "a": [[(2, 3.0), (5, 1.0)], [(7, 1.0)]],
               "b": [[], []]}
    stats = score(results, ["a", "b"])
    assert stats == {"a": [(0.5, 0)], "b": [(0.0, 0)]}


def test_example_retriever_finds_similar_latin_and_prompt_shape():
    pool = [{"src": "goy duu bn", "tgt": "гоё дуу байна"}, {"src": "margaash irne", "tgt": "маргааш ирнэ"},
            {"src": "ene kino goy", "tgt": "энэ кино гоё"}]
    ex = ExampleRetriever(pool)("goy duu bna", k=2)
    assert ex[-1]["src"] == "goy duu bn"  # most similar sits last, next to the input
    assert all(e["src"] != "margaash irne" for e in ex)
    prompt = user_prompt("goy duu bna", ex)
    assert prompt.endswith("<message>\ngoy duu bna\n</message>") and "Cyrillic: гоё дуу байна" in prompt
    assert user_prompt("x", []).startswith("Convert this message")
