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

from tatqa_loader import (  # noqa: E402
    _derivation_operands,
    _normalize_cell,
    _table_gold_ids,
    build_chunk_index,
    load_tatqa,
)

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


def test_normalize_cell_collapses_currency_formatting():
    """CLAUDE.md numeric-normalization guardrail: "$1,496.5" and "1496.50" are one value."""
    assert _normalize_cell("$1,496.5") == 1496.5
    assert _normalize_cell("1496.50") == 1496.5
    assert _normalize_cell("(1,234)") == -1234.0   # accounting negative
    assert _normalize_cell("5.2%") == 5.2
    assert _normalize_cell("  ") is None
    assert _normalize_cell("Appliances") == "appliances"


def test_derivation_operands_reads_the_cells_the_annotator_used():
    assert _derivation_operands("680-774") == [680.0, 774.0]
    assert _derivation_operands("(5,686-6,092)/6,092") == [5686.0, 6092.0]
    assert _derivation_operands(None) == []
    # "* 100" is a unit conversion in a percentage derivation, not a table cell
    assert 100.0 not in _derivation_operands("(680-774)/774 * 100")
    # count-type derivations list row labels instead of numbers
    assert "customer relationships" in _derivation_operands(
        "Customer relationships## Underlying rights and other"
    )


def test_table_gold_ids_matches_cells_not_substrings():
    """The bug cell-level matching exists to prevent: a substring test over the linearized
    row text matches "680" inside "6,801" and marks the wrong row as evidence."""
    row_cells = {
        "table_row_1": [_normalize_cell("Appliances"), 680.0, 774.0],
        "table_row_2": [_normalize_cell("Other"), 6801.0, 9000.0],
    }
    ids = _table_gold_ids(row_cells, answers=[-94], derivation="680-774")
    assert ids == ["table_row_1"], "operand 680 must not match the cell 6,801"


def test_table_gold_ids_recovers_evidence_for_computed_answers():
    """Arithmetic answers appear in no row by construction -- without the derivation operands
    these questions have no gold evidence at all and drop out of retrieval eval entirely
    (that was 474 of TAT-QA dev's 497 table-arithmetic questions)."""
    row_cells = {"table_row_1": [_normalize_cell("Appliances"), 680.0, 774.0]}
    assert _table_gold_ids(row_cells, answers=[-94], derivation=None) == []
    assert _table_gold_ids(row_cells, answers=[-94], derivation="680-774") == ["table_row_1"]


def test_most_table_questions_in_the_fixture_get_gold_evidence():
    with_gold = sum(1 for _, q in load_tatqa(str(SAMPLE)) if q.gold_chunk_ids)
    total = sum(1 for _ in load_tatqa(str(SAMPLE)))
    assert with_gold / total > 0.8, (
        f"only {with_gold}/{total} questions have gold evidence -- the table-evidence "
        f"heuristic regressed; TAT-QA retrieval eval silently loses those questions"
    )


def test_build_chunk_index_dedupes():
    all_chunks = []
    for chunks, _ in load_tatqa(str(SAMPLE)):
        all_chunks.extend(chunks)
    deduped = build_chunk_index(all_chunks)
    keys = [(c.doc_id, c.chunk_id) for c in deduped]
    assert len(keys) == len(set(keys)), "build_chunk_index left duplicate (doc_id, chunk_id) pairs"
