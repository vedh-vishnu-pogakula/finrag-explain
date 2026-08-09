"""
Smoke test for finqa_loader.py, run against the checked-in 8-example fixture
(data/sample/finqa_sample.json) so it needs no network access and runs in <1s.

Run with:  pytest tests/test_finqa_loader.py -v
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ingestion"))

from finqa_loader import load_finqa, build_chunk_index  # noqa: E402

SAMPLE = ROOT / "data" / "sample" / "finqa_sample.json"


def test_sample_file_exists():
    assert SAMPLE.exists(), "data/sample/finqa_sample.json missing -- see README setup steps"


def test_loads_expected_count():
    raw = json.loads(SAMPLE.read_text())
    results = list(load_finqa(str(SAMPLE)))
    assert len(results) == len(raw) == 8


def test_every_question_has_at_least_one_gold_chunk():
    for chunks, question in load_finqa(str(SAMPLE)):
        assert question.question, "empty question text"
        assert len(question.gold_chunk_ids) >= 1, f"{question.qa_id} has no gold_inds"


def test_gold_chunk_ids_resolve_to_real_chunks():
    """The whole point of reusing FinQA's own text_N/table_N indexing: gold_chunk_ids must
    always be found among that example's own chunks, with zero re-mapping."""
    for chunks, question in load_finqa(str(SAMPLE)):
        chunk_ids = {c.chunk_id for c in chunks}
        missing = set(question.gold_chunk_ids) - chunk_ids
        assert not missing, f"{question.qa_id}: gold ids not found in chunks: {missing}"


def test_table_row_linearization_keeps_header_value_pairing():
    """Guardrail check: table rows must NOT be flattened into one undifferentiated string --
    every value should still be tied to its column header in the linearized text."""
    all_chunks = []
    for chunks, _ in load_finqa(str(SAMPLE)):
        all_chunks.extend(chunks)
    table_chunks = [c for c in all_chunks if c.chunk_type == "table_row"]
    assert table_chunks, "no table_row chunks produced -- did the fixture change?"
    for c in table_chunks[:20]:
        assert " of " in c.text and " is " in c.text, (
            f"{c.chunk_id} doesn't look header-mapped: {c.text!r}"
        )


def test_build_chunk_index_dedupes():
    all_chunks = []
    for chunks, _ in load_finqa(str(SAMPLE)):
        all_chunks.extend(chunks)
    deduped = build_chunk_index(all_chunks)
    keys = [(c.doc_id, c.chunk_id) for c in deduped]
    assert len(keys) == len(set(keys)), "build_chunk_index left duplicate (doc_id, chunk_id) pairs"
