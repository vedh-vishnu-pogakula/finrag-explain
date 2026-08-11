"""
Smoke test for tatqa_loader.py, run against the checked-in 8-example fixture
(data/sample/tatqa_sample.json) -- mirrors test_finqa_loader.py's structure.

Run with:  pytest tests/test_tatqa_loader.py -v
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ingestion"))

from tatqa_loader import load_tatqa, build_chunk_index  # noqa: E402

SAMPLE = ROOT / "data" / "sample" / "tatqa_sample.json"


def test_sample_file_exists():
    assert SAMPLE.exists(), "data/sample/tatqa_sample.json missing -- see README setup steps"


def test_loads_expected_question_count():
    raw = json.loads(SAMPLE.read_text())
    expected = sum(len(doc["questions"]) for doc in raw)
    results = list(load_tatqa(str(SAMPLE)))
    assert len(results) == expected


def test_text_gold_ids_resolve_to_real_chunks():
    for chunks, question in load_tatqa(str(SAMPLE)):
        chunk_ids = {c.chunk_id for c in chunks}
        text_gold = [g for g in question.gold_chunk_ids if g.startswith("para_")]
        missing = set(text_gold) - chunk_ids
        assert not missing, f"{question.qa_id}: paragraph gold ids not found: {missing}"


def test_table_row_linearization_keeps_header_value_pairing():
    all_chunks = []
    for chunks, _ in load_tatqa(str(SAMPLE)):
        all_chunks.extend(chunks)
    table_chunks = [c for c in all_chunks if c.chunk_type == "table_row"]
    assert table_chunks, "no table_row chunks produced -- did the fixture change?"
    for c in table_chunks[:20]:
        assert " of " in c.text and " is " in c.text, (
            f"{c.chunk_id} doesn't look header-mapped: {c.text!r}"
        )


def test_known_answer_is_recoverable_from_its_gold_table_row():
    """Regression test for the header-merge bug found during Session 1: a table-evidence
    question's answer string should appear verbatim in at least one of its gold table_row
    chunks' linearized text."""
    found = False
    for chunks, question in load_tatqa(str(SAMPLE)):
        if question.question != "What is the amount of total sales in 2019?":
            continue
        found = True
        by_id = {c.chunk_id: c for c in chunks}
        table_gold = [g for g in question.gold_chunk_ids if g.startswith("table_row_")]
        assert table_gold, "expected at least one table_row gold id for this question"
        assert any("$1,496.5" in by_id[g].text for g in table_gold if g in by_id)
    assert found, "fixture no longer contains the expected regression-test question"


def test_build_chunk_index_dedupes():
    all_chunks = []
    for chunks, _ in load_tatqa(str(SAMPLE)):
        all_chunks.extend(chunks)
    deduped = build_chunk_index(all_chunks)
    keys = [(c.doc_id, c.chunk_id) for c in deduped]
    assert len(keys) == len(set(keys)), "build_chunk_index left duplicate (doc_id, chunk_id) pairs"
