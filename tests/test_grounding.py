"""
Tests for the Month 5 evidence-grounding layer (src/grounding/) and the RAGAS zero-cost
guard (src/faithfulness/ragas_local.py).

Run with:  pytest tests/test_grounding.py -v

Entirely offline: grounding is exercised through `StubEntailmentScorer`, and the RAGAS tests
assert the *guards* rather than running an evaluation, so no model is downloaded and no
network call is made. The billing guard in particular has to be testable without spending
money to find out whether it works.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "src" / "grounding", ROOT / "src" / "faithfulness",
           ROOT / "src" / "generation", ROOT / "src" / "ingestion",
           ROOT / "src" / "retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from entailment import (  # noqa: E402
    EntailmentResult,
    StubEntailmentScorer,
    _resolve_label_indices,
    build_scorer,
)
from grounder import Grounder  # noqa: E402
from ragas_local import (  # noqa: E402
    PaidProviderError,
    _PAID_CREDENTIAL_VARS,
    no_paid_providers,
    verify_metrics_are_local,
)
from segmenter import answer_claims, evidence_sentences, split_table_row  # noqa: E402
from staged import (  # noqa: E402
    StagedFaithfulness,
    StatementSet,
    append_statements,
    cache_is_stale,
    load_statements,
)


def _hit(chunk_id, text, chunk_type="table_row"):
    return {"chunk_id": chunk_id, "doc_id": "doc1", "chunk_type": chunk_type, "text": text}


@pytest.fixture
def retrieved():
    return [
        _hit("table_2", "company the contingent rental of 2009 is 19 ; the contingent "
                        "rental of 2008 is 22 ; the contingent rental of 2007 is 33 ;"),
        _hit("text_1", "Operating lease rental expense was 238 in 2009. The company leases "
                       "office space under non-cancellable terms.", chunk_type="text"),
    ]


# ---- segmentation -------------------------------------------------------------------------

def test_table_rows_split_on_their_own_delimiter():
    """A linearized row has no sentence boundaries a linguistic splitter can find. spaCy
    returns it as one long sentence, which then entails almost nothing because the relevant
    fact is buried among the irrelevant ones -- so rows split on the ';' the loader wrote."""
    parts = split_table_row("company the contingent rental of 2009 is 19 ; "
                            "the contingent rental of 2007 is 33 ;")
    assert len(parts) == 2
    assert parts[0].endswith("19")
    assert "2007" in parts[1]


def test_each_split_fact_keeps_its_row_label():
    """'of 2007 is 33' is not checkable on its own; 'the contingent rental of 2007 is 33' is."""
    parts = split_table_row("company the net revenue of 2009 is 680 ; of 2008 is 774 ;")
    assert "net revenue" in parts[0]


def test_evidence_sentences_keep_their_chunk_id(retrieved):
    """Grounding has to be actionable: Contribution 2 perturbs the evidence a metric relies
    on, which is only possible if a support decision points back at a specific chunk."""
    sentences = evidence_sentences(retrieved)
    assert len(sentences) > len(retrieved)          # rows and prose were both split
    assert {s.chunk_id for s in sentences} == {"table_2", "text_1"}
    assert all(s.index >= 0 for s in sentences)
    table = [s for s in sentences if s.chunk_id == "table_2"]
    assert [s.index for s in table] == list(range(len(table)))


def test_a_bare_numeric_answer_is_paired_with_its_question():
    """'-42.4' cannot be entailed or contradicted by anything. Scoring it directly would mark
    every correct numeric answer ungrounded regardless of the evidence."""
    claims = answer_claims("-42.4", question="what was the percentage change?")
    assert len(claims) == 1
    assert claims[0].kind == "text"
    assert "-42.4" in claims[0].text and "percentage change" in claims[0].text


def test_an_arithmetic_answer_is_grounded_on_its_operands_not_its_value():
    """The central design decision of this layer, and it was forced by measurement: NLI on the
    final value scores ~0.09 entailment even when the evidence is exactly right, because
    "the rental of 2009 is 19" genuinely does not entail "the change was -42.4" -- getting
    between them requires arithmetic no NLI model performs. The expression names the figures
    the answer actually consumed, and those are checkable by provenance."""
    claims = answer_claims("-42.4", question="q", expression="(19 - 33) / 33 * 100")
    assert [c.kind for c in claims] == ["numeric", "numeric"]
    assert sorted(c.value for c in claims) == [19.0, 33.0]


def test_scaffolding_constants_are_not_grounded():
    """'100' appears in every percentage expression and in plenty of tables. Counting it as
    evidence-backed would inflate the grounding rate for free."""
    claims = answer_claims("12.4", question="q", expression="(1245 - 1108) / 1108 * 100")
    assert 100.0 not in [c.value for c in claims]
    assert sorted(c.value for c in claims) == [1108.0, 1245.0]


def test_operands_are_matched_on_magnitude():
    """A table renders negatives as "(774)" and a model writes "-774"; and in "680-774" the
    minus is an operator, not a sign."""
    claims = answer_claims("-94", question="q", expression="680 - 774")
    assert sorted(c.value for c in claims) == [680.0, 774.0]


def test_a_long_prose_answer_is_split_into_claims():
    answer = ("Revenue increased to 680 million in 2009. The increase was driven by higher "
              "volumes. Margins were flat.")
    assert len(answer_claims(answer)) >= 2


def test_empty_answer_produces_no_claims():
    assert answer_claims("") == []
    assert answer_claims(None) == []


# ---- entailment ---------------------------------------------------------------------------

def test_nli_label_order_is_read_from_the_checkpoint_not_assumed():
    """The trap this guards: checkpoints disagree on label order. deberta-large-mnli is
    (contradiction, neutral, entailment) while several community models are (contradiction,
    entailment, neutral). Hard-coding index 2 -- which most example code does -- silently
    returns the *neutral* probability for half the Hub."""
    a = _resolve_label_indices({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"})
    assert a == (2, 0, 1)
    b = _resolve_label_indices({0: "contradiction", 1: "entailment", 2: "neutral"})
    assert b == (1, 0, 2)


def test_uninterpretable_labels_raise_rather_than_guess():
    """A wrong index does not fail loudly -- it returns a plausible number for the wrong
    class and every downstream grounding figure is quietly wrong."""
    with pytest.raises(ValueError, match="cannot identify NLI labels"):
        _resolve_label_indices({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_stub_scorer_runs_offline(retrieved):
    scores = StubEntailmentScorer().score_pairs([("the revenue was 680", "revenue 680")])
    assert isinstance(scores[0], EntailmentResult)
    assert 0.0 <= scores[0].entailment <= 1.0


def test_build_scorer_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown grounding backend"):
        build_scorer({"grounding": {}}, backend="gpt4")


# ---- grounder -----------------------------------------------------------------------------

def test_grounding_picks_the_best_sentence_not_the_average(retrieved):
    """A claim is grounded if *some* sentence entails it. Averaging across all retrieved
    sentences would punish a perfectly grounded answer for the irrelevant chunks that came
    back alongside the right one -- which at top-k=5 is the normal case."""
    grounder = Grounder(StubEntailmentScorer())
    result = grounder.ground("the contingent rental of 2007 is 33",
                             retrieved, question="what was it?")
    assert result.n_claims == 1
    assert result.n_evidence_sentences > 2
    best = result.support[0]
    assert best.chunk_id == "table_2"
    assert "2007" in best.sentence
    # the aggregate is the best match, not diluted by the other sentences
    assert result.groundedness == pytest.approx(best.entailment)


def test_grounded_chunk_ids_are_distinct_from_self_reported_citations(retrieved):
    """The gap between what a model *says* it used and what its answer is actually entailed by
    is a finding in its own right, so the two must never be conflated."""
    grounder = Grounder(StubEntailmentScorer(), support_threshold=0.1)
    result = grounder.ground("the contingent rental of 2007 is 33", retrieved, question="q")
    assert result.cited_chunk_ids == ["table_2"]


def test_numeric_grounding_locates_the_sentence_supplying_the_operand(retrieved):
    """Provenance, and it points at a specific (chunk, sentence). Contribution 2 perturbs the
    evidence a metric relies on, so a support decision that only named a chunk would be too
    coarse to remove one fact at a time."""
    result = Grounder(StubEntailmentScorer()).ground(
        "-14", retrieved, question="q", expression="(19 - 33)")
    assert [s.kind for s in result.support] == ["numeric", "numeric"]
    assert all(s.supported for s in result.support)
    assert result.cited_chunk_ids == ["table_2"]
    assert "19" in result.support[0].sentence
    assert result.support[0].sentence_index is not None


def test_an_operand_absent_from_the_evidence_is_not_grounded(retrieved):
    """The hallucinated-operand case: the model computed with a figure nobody supplied."""
    result = Grounder(StubEntailmentScorer()).ground(
        "999", retrieved, question="q", expression="(88888 - 77777)")
    assert not any(s.supported for s in result.support)
    assert result.cited_chunk_ids == []
    assert result.supported_rate == 0.0


def test_numeric_provenance_needs_no_entailment_model(retrieved):
    """On FinQA almost every answer is arithmetic, so this keeps the NLI model off the
    critical path entirely -- grounding a full run costs a lookup, not a forward pass."""
    class _Exploding(StubEntailmentScorer):
        def score_pairs(self, pairs):
            assert not pairs, "numeric claims must never reach the entailment model"
            return []

    result = Grounder(_Exploding()).ground("-14", retrieved, question="q",
                                           expression="(19 - 33)")
    assert result.supported_rate == 1.0


def test_batch_grounding_collects_every_pair_before_scoring(retrieved):
    """Scoring per question would turn one batched pass into hundreds of tiny ones -- the same
    anti-pattern the attribution engine avoids by pre-collecting coalitions."""
    calls = []

    class _Counting(StubEntailmentScorer):
        def score_pairs(self, pairs):
            calls.append(len(pairs))
            return super().score_pairs(pairs)

    grounder = Grounder(_Counting())
    results = grounder.ground_batch([("33", retrieved, "q1"), ("19", retrieved, "q2")])
    assert len(results) == 2
    assert len(calls) == 1, "batch grounding must issue exactly one scoring pass"


def test_contradiction_is_tracked_separately_from_support(retrieved):
    """An answer contradicted by its evidence and one merely unsupported by it are different
    failures -- hallucination against present evidence vs a retrieval gap."""
    class _Contradicting(StubEntailmentScorer):
        def score_pairs(self, pairs):
            return [EntailmentResult(entailment=0.1, contradiction=0.9, neutral=0.0)
                    for _ in pairs]

    result = Grounder(_Contradicting()).ground("revenue was 999", retrieved, question="q")
    assert result.support[0].supported is False
    assert result.support[0].contradicted is True
    assert result.contradicted_rate == 1.0


def test_a_supported_claim_is_never_reported_as_contradicted(retrieved):
    """Retrieved evidence holds many facts, and an NLI model reads "revenue 2018 was 440.7"
    as contradicting a claim about 2019. Taking the raw maximum across all sentences fires on
    ordinary multi-year tables -- it reported 44% of TAT-QA answers as contradicted, which
    says nothing about faithfulness. Contradiction only counts when nothing supported the
    claim, which is the question the field exists to answer."""
    class _SupportedButAlsoContradicted(StubEntailmentScorer):
        def score_pairs(self, pairs):
            return [EntailmentResult(entailment=0.95, contradiction=0.9, neutral=0.0)
                    for _ in pairs]

    result = Grounder(_SupportedButAlsoContradicted()).ground("x", retrieved, question="q")
    assert result.support[0].supported is True
    assert result.support[0].contradicted is False
    assert result.support[0].contradiction == pytest.approx(0.9), "raw maximum still recorded"


def test_grounding_handles_an_answer_with_no_evidence():
    result = Grounder(StubEntailmentScorer()).ground("42", [], question="q")
    assert result.n_evidence_sentences == 0
    assert result.groundedness == 0.0
    assert result.cited_chunk_ids == []


# ---- RAGAS zero-cost guard ----------------------------------------------------------------

def test_paid_credentials_are_neutralised_inside_the_guard(monkeypatch):
    """RAGAS's own factory signature is llm_factory(model, provider="openai"). If some
    internal still reaches for OpenAI it must get a 401, not a working key and an invoice."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-a-real-key-that-would-bill")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    with no_paid_providers():
        assert os.environ["OPENAI_API_KEY"].startswith("sk-LOCAL-JUDGE-ONLY")
        assert "ANTHROPIC_API_KEY" not in os.environ
        assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
    assert os.environ["OPENAI_API_KEY"] == "sk-a-real-key-that-would-bill"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-real"


def test_the_guard_restores_the_environment_even_when_the_run_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    with pytest.raises(RuntimeError):
        with no_paid_providers():
            raise RuntimeError("evaluation blew up")
    assert os.environ["OPENAI_API_KEY"] == "sk-real"


def test_every_known_paid_credential_is_covered():
    """A new provider added to langchain is a new way to bill silently."""
    for name in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "GOOGLE_API_KEY", "COHERE_API_KEY", "GROQ_API_KEY"):
        assert name in _PAID_CREDENTIAL_VARS


def test_a_metric_without_a_judge_is_refused_before_evaluation():
    """Checked before evaluate(), because 'after' means the charge already happened. llm=None
    is the exact failure mode: RAGAS fills it in from its own default, and that default is
    OpenAI."""
    class _Unconfigured:
        name = "faithfulness"
        llm = None

    with pytest.raises(PaidProviderError, match="OpenAI default"):
        verify_metrics_are_local([_Unconfigured()])


def test_a_metric_needing_no_llm_is_allowed_through():
    class _NoLLMMetric:
        name = "context_precision_nonllm"

    verify_metrics_are_local([_NoLLMMetric()])      # must not raise


# ---- staged faithfulness (the Contribution 2 engine) ---------------------------------------

class _FakeVerifier:
    """Stands in for the NLI verifier: a statement is supported iff its final token appears in
    the premise. Keeps the whole test suite offline -- injecting `_verifier` rather than a
    fake metric matters, because verify() reads the verifier directly and a fixture that only
    patched the metric would silently download a real model."""

    model_name = "fake-verifier"

    def score_pairs(self, pairs):
        return [EntailmentResult(entailment=1.0 if s.split()[-1] in p else 0.0,
                                 contradiction=0.0, neutral=0.0)
                for p, s in pairs]


def _staged():
    scorer = StagedFaithfulness(llm=None)
    scorer._verifier = _FakeVerifier()
    return scorer


def test_verify_reruns_without_any_llm():
    """The property Month 6 depends on. Stage 1 (decompose) needs the judge; stage 2 (verify)
    must not, or a perturbation sweep costs thousands of LLM calls recomputing a result that
    cannot have changed."""
    scorer = _staged()
    assert scorer.llm is None          # no judge configured at all
    result = scorer.verify(["rental 2009 is 19"], ["the contingent rental of 2009 is 19"])
    assert result.score == 1.0


def test_removing_the_supporting_evidence_moves_the_score_and_names_the_claim():
    """Contribution 2 in miniature. A score that fell from 1.0 to 0.5 without saying *which*
    statement lost its support is not evidence of anything."""
    scorer = _staged()
    contexts = ["the contingent rental of 2009 is 19", "the net rental expense is 257"]
    statements = ["rental 2009 is 19", "expense is 257"]

    before = scorer.verify(statements, contexts)
    after = scorer.verify(statements, contexts[1:])          # drop the first fact

    assert before.score == 1.0
    assert after.score == 0.5
    assert after.verdicts[0]["supported"] is False
    assert after.verdicts[1]["supported"] is True


def test_an_answer_with_no_statements_scores_nan_not_zero():
    """A declined answer has nothing to be unfaithful about. Scoring it 0.0 would drag the
    dataset mean down and misreport abstention as hallucination -- and the 7B model declines
    on 10% of TAT-QA."""
    result = _staged().verify([], ["some context"])
    assert result.n_statements == 0
    assert result.score != result.score                       # NaN
    assert result.to_json()["faithfulness"] is None


def test_statement_cache_round_trips_and_survives_a_torn_line(tmp_path):
    path = tmp_path / "statements.jsonl"
    append_statements(path, [StatementSet("q1", "Q1", "-14", ["a", "b"]),
                             StatementSet("q2", "Q2", "5", ["c"])])
    with open(path, "a") as handle:
        handle.write('{"qa_id": "q3", "statem')          # killed mid-write

    cache = load_statements(path)
    assert set(cache) == {"q1", "q2"}
    assert cache["q1"].statements == ["a", "b"]


def test_a_cache_entry_from_a_different_answer_is_detected_as_stale():
    """The one way this cache corrupts results silently: B1 is re-run, answers change, and
    stale statements get verified against the new contexts. The numbers stay plausible."""
    entry = StatementSet("q1", "Q", "-14", ["the change was -14"])
    assert cache_is_stale(entry, "-42.4") is True
    assert cache_is_stale(entry, "-14") is False
    assert cache_is_stale(entry, " -14 ") is False           # whitespace is not a change
