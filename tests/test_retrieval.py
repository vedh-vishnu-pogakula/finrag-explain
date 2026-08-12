"""
Tests for the Month 3 retrieval layer (src/retrieval/) and eval/metrics/retrieval_metrics.py.

Run with:  pytest tests/test_retrieval.py -v

Like the loader tests these run offline in ~1s: the default embedder here is `HashEmbedder`,
the deterministic stub, so nothing downloads a model. One test exercises the real
sentence-transformers path (tokenizer-based truncation, which is a stated project guardrail
and worth testing against the actual tokenizer) and skips itself if the model isn't available
locally.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "src" / "ingestion", ROOT / "src" / "retrieval",
           ROOT / "eval" / "metrics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from chunk_index import ChunkIndex, RetrievedChunk  # noqa: E402
from embedder import Embedder, HashEmbedder  # noqa: E402
from finqa_loader import build_chunk_index, load_finqa  # noqa: E402
from retrieval_metrics import evaluate_one, gold_type_of, summarize  # noqa: E402
from retriever import Retriever  # noqa: E402
from schema import Chunk, Question, read_chunks, write_jsonl  # noqa: E402

FINQA_SAMPLE = ROOT / "data" / "sample" / "finqa_sample.json"


@pytest.fixture(scope="module")
def sample_data():
    """Real chunks from the checked-in FinQA fixture -- the same data the loader tests use."""
    all_chunks, questions = [], []
    for chunks, question in load_finqa(str(FINQA_SAMPLE)):
        all_chunks.extend(chunks)
        questions.append(question)
    return build_chunk_index(all_chunks), questions


@pytest.fixture(scope="module")
def index(sample_data):
    chunks, _ = sample_data
    return ChunkIndex.build(chunks, HashEmbedder())


# ---- embedder ---------------------------------------------------------------------------

def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder()
    a = e.encode_chunks(["net revenue increased by 5.2%"])
    b = e.encode_chunks(["net revenue increased by 5.2%"])
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0), "embeddings must be L2-normalized"


def test_empty_input_returns_empty_matrix_not_a_crash():
    e = HashEmbedder()
    assert e.encode_chunks([]).shape == (0, e.dim)


@pytest.mark.slow
def test_real_tokenizer_enforces_token_cap():
    """CLAUDE.md guardrail: the chunk length cap must come from the embedding model's own
    tokenizer, not word/char counts. A long linearized table row is the case that matters --
    digits and commas tokenize much denser than prose."""
    pytest.importorskip("sentence_transformers")
    e = Embedder(max_tokens=32)
    try:
        e.model
    except Exception as exc:  # no local model + no network
        pytest.skip(f"embedding model unavailable: {exc}")

    row = "company " + " ".join(
        f"the net revenue of {2000 + i} is {i},{i}{i}{i} ;" for i in range(40)
    )
    assert e.count_tokens(row) > 32, "fixture is too short to test truncation"
    truncated = e.truncate(row)
    assert e.count_tokens(truncated) <= 30  # cap minus the two special tokens
    assert e.truncate("short row") == "short row", "short text must pass through untouched"

    report = e.truncation_report([row, "short row"])
    assert report["n_truncated"] == 1 and report["n_texts"] == 2


def test_max_tokens_above_model_limit_is_rejected():
    pytest.importorskip("sentence_transformers")
    e = Embedder(max_tokens=99_999)
    try:
        with pytest.raises(ValueError, match="max_seq_length"):
            e.model
    except OSError as exc:
        pytest.skip(f"embedding model unavailable: {exc}")


# ---- index ------------------------------------------------------------------------------

def test_index_build_shapes(index, sample_data):
    chunks, _ = sample_data
    assert len(index) == len(chunks)
    assert index.embeddings.shape == (len(chunks), HashEmbedder().dim)
    assert index.meta["n_docs"] == len(set(c.doc_id for c in chunks))


def test_chunk_id_collisions_across_docs_stay_separated(index, sample_data):
    """FinQA reuses chunk_ids in every document (`text_3` exists in all of them). Per-doc
    scoping must key on (doc_id, chunk_id) -- if it didn't, one document's chunks would leak
    into another's candidate pool and the gold-id comparison would silently score the wrong
    rows."""
    chunks, _ = sample_data
    ids_per_doc = {}
    for c in chunks:
        ids_per_doc.setdefault(c.doc_id, set()).add(c.chunk_id)
    docs = list(ids_per_doc)
    assert len(docs) > 1
    assert ids_per_doc[docs[0]] & ids_per_doc[docs[1]], "fixture should have colliding ids"

    for doc_id in docs:
        for c in index.chunks_for_doc(doc_id):
            assert c.doc_id == doc_id


def test_score_matrix_is_the_search_ranking(index, sample_data):
    """search() must be exactly argsort(score()) -- Month 4 attribution reads score()
    directly and its deltas are only meaningful if that's the retriever's real ranking."""
    _, questions = sample_data
    q = questions[0]
    e = HashEmbedder()
    qvec = e.encode_queries([q.question])

    scores = index.score(qvec, doc_id=q.doc_id)[0]
    candidates = index.chunks_for_doc(q.doc_id)
    expected = [candidates[i].chunk_id for i in np.argsort(-scores)[:5]]

    hits = index.search(qvec, k=5, doc_id=q.doc_id)[0]
    assert [h.chunk.chunk_id for h in hits] == expected
    assert [h.rank for h in hits] == [0, 1, 2, 3, 4]
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_score_handles_a_batch_of_query_variants(index, sample_data):
    """The attribution primitive: N perturbed queries scored against one doc in a single
    matmul -> (N, n_candidates)."""
    _, questions = sample_data
    q = questions[0]
    variants = [q.question, q.question.replace("net", ""), "unrelated text"]
    scores = index.score(HashEmbedder().encode_queries(variants), doc_id=q.doc_id)
    assert scores.shape == (3, len(index.chunks_for_doc(q.doc_id)))


def test_corpus_scope_search_uses_faiss(index):
    pytest.importorskip("faiss")
    qvec = HashEmbedder().encode_queries(["net revenue"])
    hits = index.search(qvec, k=3, doc_id=None)[0]
    assert len(hits) == 3
    # exact flat-IP search must agree with the brute-force matmul it's meant to replace
    brute = index.score(qvec)[0]
    assert hits[0].chunk.chunk_id == index.chunks[int(np.argmax(brute))].chunk_id


def test_index_save_load_round_trip(index, tmp_path):
    index.save(tmp_path)
    reloaded = ChunkIndex.load(tmp_path, expect_model=HashEmbedder.model_name)
    assert len(reloaded) == len(index)
    assert np.allclose(reloaded.embeddings, index.embeddings)
    assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in index.chunks]
    assert reloaded.chunks[0].doc_id == index.chunks[0].doc_id


def test_load_rejects_a_different_embedding_model(index, tmp_path):
    """Querying with model A against chunk vectors built by model B produces in-range scores
    and nonsense rankings -- the kind of bug that costs a week. Fail loudly instead."""
    index.save(tmp_path)
    with pytest.raises(ValueError, match="rebuild the index"):
        ChunkIndex.load(tmp_path, expect_model="BAAI/bge-small-en-v1.5")


def test_index_rejects_mismatched_inputs(sample_data):
    chunks, _ = sample_data
    with pytest.raises(ValueError, match="length mismatch"):
        ChunkIndex(chunks, np.zeros((len(chunks) - 1, 8), dtype=np.float32))


def test_unknown_doc_id_raises(index):
    with pytest.raises(KeyError):
        index.rows_for_doc("no/such/doc.pdf-1")


# ---- retriever --------------------------------------------------------------------------

def test_retriever_returns_top_k_from_the_right_document(index, sample_data):
    _, questions = sample_data
    r = Retriever(embedder=HashEmbedder(), index=index, top_k=5, scope="document")
    q = questions[0]
    hits = r.retrieve(q.question, doc_id=q.doc_id)
    assert len(hits) == 5
    assert all(isinstance(h, RetrievedChunk) and h.chunk.doc_id == q.doc_id for h in hits)


def test_retriever_batch_matches_single_calls(index, sample_data):
    _, questions = sample_data
    r = Retriever(embedder=HashEmbedder(), index=index, top_k=3, scope="document")
    qs = questions[:4]
    batched = r.retrieve_batch([q.question for q in qs], [q.doc_id for q in qs])
    for q, hits in zip(qs, batched):
        single = r.retrieve(q.question, doc_id=q.doc_id)
        assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in single]


def test_document_scope_without_doc_id_is_an_error(index):
    r = Retriever(embedder=HashEmbedder(), index=index, top_k=3, scope="document")
    with pytest.raises(ValueError, match="needs a doc_id"):
        r.retrieve("net revenue")


def test_k_larger_than_the_document_is_clamped(index, sample_data):
    _, questions = sample_data
    q = questions[0]
    n = len(index.chunks_for_doc(q.doc_id))
    r = Retriever(embedder=HashEmbedder(), index=index, top_k=n + 50, scope="document")
    assert len(r.retrieve(q.question, doc_id=q.doc_id)) == n


def test_score_query_variants_returns_aligned_candidates(index, sample_data):
    _, questions = sample_data
    q = questions[0]
    r = Retriever(embedder=HashEmbedder(), index=index, scope="document")
    scores, candidates = r.score_query_variants([q.question, "x"], doc_id=q.doc_id)
    assert scores.shape == (2, len(candidates))
    assert [c.chunk_id for c in candidates] == [
        c.chunk_id for c in index.chunks_for_doc(q.doc_id)
    ]


def test_gold_chunks_are_retrievable_at_all(index, sample_data):
    """Sanity, not quality: with a bag-of-words stub embedder, at least some questions in the
    fixture should surface their gold evidence in the top 5 of their own document. If this
    goes to zero, chunk_ids and gold_inds have drifted apart -- a wiring bug, not a retrieval
    result."""
    _, questions = sample_data
    r = Retriever(embedder=HashEmbedder(), index=index, top_k=5, scope="document")
    hit = sum(
        1 for q in questions
        if set(q.gold_chunk_ids) & {h.chunk.chunk_id for h in r.retrieve(q.question, q.doc_id)}
    )
    assert hit > 0, "no gold chunk retrieved for any question -- chunk_id/gold_id mismatch?"


# ---- schema jsonl round-trip ------------------------------------------------------------

def test_chunks_round_trip_through_jsonl(sample_data, tmp_path):
    chunks, _ = sample_data
    path = write_jsonl(chunks[:20], tmp_path / "chunks.jsonl")
    reloaded = read_chunks(path)
    assert [c.to_json() for c in reloaded] == [c.to_json() for c in chunks[:20]]


# ---- metrics ----------------------------------------------------------------------------

def test_evaluate_one_known_values():
    m = evaluate_one(["a", "b", "c", "d", "e"], ["b", "z"], ks=[1, 3, 5])
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["recall@3"] == 0.5           # 1 of 2 gold facts
    assert m["precision@3"] == pytest.approx(1 / 3)
    assert m["full_recall@5"] == 0.0      # "z" never retrieved
    assert m["mrr"] == 0.5                # gold "b" at rank 2
    assert m["n_gold"] == 2


def test_full_recall_requires_every_gold_chunk():
    m = evaluate_one(["a", "b"], ["a", "b"], ks=[2])
    assert m["full_recall@2"] == 1.0 and m["recall@2"] == 1.0


def test_evaluate_one_with_no_gold_is_empty():
    assert evaluate_one(["a"], [], ks=[1]) == {}


def test_summarize_excludes_but_counts_ungraded_questions():
    records = [
        {"qa_id": "1", "dataset": "finqa", "gold_type": "table",
         "metrics": evaluate_one(["a"], ["a"], ks=[1])},
        {"qa_id": "2", "dataset": "finqa", "gold_type": "text",
         "metrics": evaluate_one(["b"], ["c"], ks=[1])},
        {"qa_id": "3", "dataset": "finqa", "gold_type": "unknown", "metrics": {}},
    ]
    report = summarize(records, ks=[1])
    assert report["n_questions_total"] == 3
    assert report["n_questions_scored"] == 2
    assert report["n_questions_skipped_no_gold"] == 1
    assert report["overall"]["recall@1"] == 0.5
    assert report["by_gold_type"]["table"]["recall@1"] == 1.0
    assert report["by_gold_type"]["text"]["recall@1"] == 0.0


def test_gold_type_of_classifies_evidence_mix():
    types = {"text_1": "text", "table_2": "table_row"}
    assert gold_type_of(["text_1"], types) == "text"
    assert gold_type_of(["table_2"], types) == "table"
    assert gold_type_of(["text_1", "table_2"], types) == "mixed"
    assert gold_type_of(["missing_9"], types) == "unknown"
