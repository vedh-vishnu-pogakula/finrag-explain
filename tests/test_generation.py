"""
Tests for the Month 3 generation layer (src/generation/) and eval/metrics/answer_metrics.py.

Run with:  pytest tests/test_generation.py -v

No API calls and no API key: the generator is exercised through `StubGenerator` and through a
fake client that returns canned Messages API responses. What's actually being tested on the
API path is the parts that go wrong silently in production -- refusal handling, truncated
JSON, and citation indices that don't correspond to anything retrieved.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "src" / "ingestion", ROOT / "src" / "retrieval",
           ROOT / "src" / "generation", ROOT / "eval" / "metrics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from answer_metrics import (  # noqa: E402
    aggregate_answers,
    exact_match,
    execution_match,
    normalize_number,
    normalize_text,
    numeric_match,
    score_answer,
    token_f1,
)
from calculator import evaluate, is_program  # noqa: E402
from chunk_index import RetrievedChunk  # noqa: E402
from generator import ApiGenerator, LocalGenerator, StubGenerator, _extract_json  # noqa: E402
from prompts import (  # noqa: E402
    ANSWER_SCHEMA,
    build_messages,
    build_user_message,
    prompt_version,
)
from schema import Chunk  # noqa: E402


def _hit(chunk_id, text, rank=0, chunk_type="table_row"):
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="doc1", dataset="finqa",
                    chunk_type=chunk_type, text=text),
        score=0.9 - 0.1 * rank,
        rank=rank,
    )


@pytest.fixture
def retrieved():
    return [
        _hit("table_2", "company the net revenue of 2009 is 680 ;", rank=0),
        _hit("table_3", "company the net revenue of 2008 is 774 ;", rank=1),
        _hit("text_5", "Net revenue declined year over year.", rank=2, chunk_type="text"),
    ]


# ---- numeric normalization (CLAUDE.md guardrail) ----------------------------------------

def test_normalize_number_handles_financial_formatting():
    assert normalize_number("$1,496.5") == 1496.5
    assert normalize_number("1496.50") == 1496.5
    assert normalize_number("(1,234)") == -1234.0       # accounting negative
    assert normalize_number("5.2%") == 5.2
    assert normalize_number("14.1") == 14.1
    assert normalize_number("increased slightly") is None
    assert normalize_number("") is None


def test_numeric_match_ignores_formatting():
    assert numeric_match("$1,496.5", "1496.50")
    assert numeric_match("(774)", "-774")
    assert numeric_match("14.06", "14.1")               # within the 1% rounding tolerance
    assert not numeric_match("680", "774")


def test_percent_equivalence_only_applies_to_percentages():
    """"5.2%" and "0.052" are the same answer; "5.2" and "0.052" dollars are not. Applying the
    x100 relation unconditionally would silently mark wrong answers correct."""
    assert numeric_match("5.2%", "0.052")
    assert numeric_match("0.052", "5.2", scale="percent")
    assert not numeric_match("5.2", "0.052")


def test_scale_field_is_not_dropped():
    """TAT-QA's `scale` changes what a bare number means -- -94 with scale "million" is
    -94,000,000, so both spellings of the same answer must score as correct."""
    assert numeric_match("-94", "-94", scale="million")
    assert numeric_match("-94000000", "-94", scale="million")
    assert not numeric_match("94", "-94", scale="million"), "a sign error is a real error"


def test_execution_accuracy_is_finqas_official_metric():
    """FinQA's display string is rounded for presentation while `exe_ans` carries the executed
    value. Scoring against the display string marks a model wrong for being *more* accurate
    than the annotation: -6.8528 fails a 1% tolerance against "-7%"."""
    assert execution_match("-6.8528", "-0.06853", is_percent=True)
    assert execution_match("10.745", "0.10745", is_percent=True)
    assert execution_match("127.4", "127.4")
    assert execution_match("yes", "yes")          # boolean gold compares as text
    assert not execution_match("1.6", "yes")
    assert not execution_match("5", None)


def test_percent_relation_is_gated_not_universal():
    """FinQA stores percentages as fractions and the prompt asks for percent form, so a factor
    of 100 is a unit convention -- but only on percentage questions. Accepting it everywhere
    would score an answer that is wrong by 100x as correct."""
    assert execution_match("5.2", "0.052", is_percent=True)
    assert not execution_match("5.2", "0.052", is_percent=False)


def test_score_answer_prefers_the_executed_gold_and_keeps_both_verdicts():
    """The two verdicts must stay separately visible: changing which one is headline is a
    change of definition, and it has to remain auditable rather than silent."""
    scores = score_answer("-6.8528", "-7%", exe_answer="-0.06853", answer_is_percent=True)
    assert scores["numeric_match"] == 1.0      # correct under execution accuracy
    assert scores["display_match"] == 0.0      # would have been wrong against the display string
    assert scores["execution_match"] == 1.0
    assert scores["is_numeric"] == 1.0


def test_empty_display_gold_is_still_scoreable_via_exe_ans():
    """12 of FinQA's 883 dev questions ship `answer` present-but-empty. Those were scored
    against "" -- unscoreable, and silently counted as span questions, which is what dragged
    FinQA's span F1 to 0.04 on a dataset that is essentially all numeric."""
    scores = score_answer("22.742", "", exe_answer="0.22742", answer_is_percent=True)
    assert scores["is_numeric"] == 1.0
    assert scores["numeric_match"] == 1.0


def test_normalize_text_drops_sentence_punctuation_but_keeps_it_inside_numbers():
    """A gold span lifted from a document ends in a full stop and the prediction does not.
    Keeping '.' everywhere cost an exact match and a token of F1 on word-for-word correct
    answers; stripping it everywhere would break decimals."""
    assert normalize_text("lower level of R&D grants.") == normalize_text("lower level of R&D grants")
    assert exact_match("the total number of days", "the total number of days.")
    assert normalize_text("1.5") == "1.5"          # decimal survives
    assert normalize_text("4.35%") == "4.35%"      # percent survives
    assert normalize_text("173.") == "173"         # trailing stop on a number is dropped


def test_text_metrics():
    assert exact_match("The Company", "the company")
    assert token_f1("net revenue increased", "revenue increased") == pytest.approx(0.8)
    assert token_f1("", "something") == 0.0


def test_score_answer_flags_which_metric_applies():
    numeric = score_answer("1,496.5", "1496.5")
    assert numeric["is_numeric"] == 1.0 and numeric["numeric_match"] == 1.0
    span = score_answer("Automotive", "Automotive", answer_type="span")
    assert span["is_numeric"] == 0.0 and span["exact_match"] == 1.0


def test_aggregate_separates_numeric_from_span_questions():
    """Averaging numeric accuracy over span questions (which never parse as floats) would drag
    the headline number down for reasons that have nothing to do with the model."""
    scores = [
        score_answer("100", "100"),                          # numeric, right
        score_answer("50", "100"),                           # numeric, wrong
        score_answer("Automotive", "Automotive"),            # span, right
    ]
    agg = aggregate_answers(scores)
    assert agg["n_numeric"] == 2 and agg["n_textual"] == 1
    assert agg["numeric_accuracy"] == 0.5
    assert agg["span_exact_match"] == 1.0


# ---- prompt ------------------------------------------------------------------------------

def test_prompt_numbers_evidence_for_citation(retrieved):
    msg = build_user_message("what was the change in net revenue?", retrieved)
    assert "[1] (table row) company the net revenue of 2009 is 680 ;" in msg
    assert "[3] (text) Net revenue declined year over year." in msg
    assert "what was the change in net revenue?" in msg


def test_answer_schema_is_strict():
    """A loose schema defeats the point -- the eval loop reads these fields positionally."""
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert set(ANSWER_SCHEMA["required"]) == {
        "answer", "answer_expression", "evidence_numbers", "insufficient_evidence"
    }


def test_exemplars_are_marked_as_non_evidence():
    """Regression guard for a real contamination bug. Exemplars used to be sent as ordinary
    user/assistant turns, structurally indistinguishable from the real question; the 1.5B model
    then answered a live question with `-30584 / 8920 * 100`, where 8920 came only from the
    span exemplar. Exactly one message may carry evidence."""
    msgs = build_messages("what was the change?", [_hit("t1", "revenue of 2009 is 680 ;")],
                          use_program=True, n_shot=3)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "NOT evidence" in msgs[0]["content"]
    assert "Worked examples" in msgs[0]["content"]
    # The exemplar figure must not appear anywhere in the turn that carries real evidence.
    assert "8920" not in msgs[1]["content"]
    assert "8920" in msgs[0]["content"], "the exemplar itself should still be present"


def test_zero_shot_prompt_has_no_exemplars():
    msgs = build_messages("q", [_hit("t1", "x")], use_program=True, n_shot=0)
    assert "Worked examples" not in msgs[0]["content"]


def test_prompt_version_encodes_the_ablation_arm():
    """Results files are matched to arms by this tag alone, so it has to be faithful."""
    assert prompt_version(use_program=True, n_shot=3) == "b1-pot-3shot-v2"
    assert prompt_version(use_program=False, n_shot=3) == "b1-direct-3shot-v2"
    # The zero-shot direct arm reproduces the original baseline and keeps its original tag.
    assert prompt_version(use_program=False, n_shot=0) == "b1-evidence-bound-v1"


# ---- calculator: the program-of-thought executor -----------------------------------------

def test_calculator_evaluates_financial_expressions():
    assert evaluate("(1245 - 1108) / 1108 * 100") == pytest.approx(12.3646, abs=1e-3)
    assert evaluate("$135.02 - $148.92") == pytest.approx(-13.90, abs=1e-6)
    assert evaluate("(1,234) - 234") == pytest.approx(1000.0)
    assert evaluate("1500 * 12%") == pytest.approx(180.0)


def test_calculator_truncates_at_the_models_own_arithmetic():
    """Models write `(a - b) / b = 0.42`. The left side is the program; the right side is the
    mental arithmetic this module exists to replace, so it is discarded."""
    assert evaluate("$ 1429 / $ 144535 * 100 = 1%") == pytest.approx(0.98869, abs=1e-4)


def test_calculator_understands_finqas_own_program_dsl():
    """FinQA ships gold programs in this notation and the model imitates it."""
    assert evaluate("divide(637, const_5)") == pytest.approx(127.4)
    assert evaluate("subtract(206588, 181001)") == pytest.approx(25587.0)
    # Nested calls are the common case -- FinQA's multi-step programs are all nested.
    assert evaluate("multiply(divide(subtract(19, 33), 33), const_100)") == pytest.approx(
        -42.4242, abs=1e-3)


@pytest.mark.parametrize("hostile", [
    "__import__('os').system('rm -rf /')",
    "open('/etc/passwd').read()",
    "[i for i in range(10)]",
    "x + 1",
    "'abc' * 3",
])
def test_calculator_rejects_anything_that_is_not_arithmetic(hostile):
    """The expression is untrusted model output, so the AST walk whitelists operators rather
    than blacklisting syntax -- there is no eval() for a payload to reach."""
    assert evaluate(hostile) is None


def test_calculator_rejects_the_denial_of_service_cases():
    assert evaluate("9**9**9") is None      # exponent cap
    assert evaluate("1/0") is None          # None, not inf: malformed, not merely wrong
    assert evaluate("x" * 500) is None


def test_is_program_does_not_treat_a_percent_literal_as_a_calculation():
    """`answer_expression` is often just the answer restated. Executing "4.35%" would turn a
    correct answer into 0.0435, so the percent rewrite must not be mistaken for a division."""
    assert is_program("4.35%") is False
    assert is_program("127.4") is False
    assert is_program("-774") is False
    assert is_program("") is False
    assert is_program("1245 - 1108") is True
    assert is_program("1500 * 12%") is True


# ---- generator: API response handling ----------------------------------------------------

@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 20


class _FakeMessages:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        self.last_request = kwargs
        return self._response


class _FakeClient:
    """Stands in for anthropic.Anthropic. `beta.messages` is wired to the same canned response
    so the refusal-fallback path is exercised by default, as it is in real runs."""

    def __init__(self, response):
        self.messages = _FakeMessages(response)
        self.beta = type("Beta", (), {"messages": _FakeMessages(response)})()


def _response(text, stop_reason="end_turn", model="claude-opus-5"):
    return type("Resp", (), {
        "content": [_Block(text)], "stop_reason": stop_reason,
        "model": model, "usage": _Usage(),
    })()


def test_generate_parses_structured_answer(retrieved):
    payload = json.dumps({"answer": "-94", "evidence_numbers": [1, 2],
                          "insufficient_evidence": False})
    gen = ApiGenerator(client=_FakeClient(_response(payload)))
    result = gen.generate("what was the change in net revenue?", retrieved)
    assert result.answer == "-94"
    assert result.cited_chunk_ids == ["table_2", "table_3"]
    assert result.insufficient_evidence is False
    assert result.error is None


def test_out_of_range_citations_are_dropped_not_repaired(retrieved):
    """A citation index pointing at nothing is a signal worth preserving. Snapping it to a real
    chunk would manufacture grounding the model never actually claimed."""
    payload = json.dumps({"answer": "5", "evidence_numbers": [1, 9, 0, -1],
                          "insufficient_evidence": False})
    gen = ApiGenerator(client=_FakeClient(_response(payload)))
    assert gen.generate("q", retrieved).cited_chunk_ids == ["table_2"]


def test_refusal_is_handled_before_reading_content(retrieved):
    """A refusal comes back as HTTP 200 with empty content -- indexing content[0] blindly is
    exactly how this breaks mid-run."""
    resp = type("Resp", (), {"content": [], "stop_reason": "refusal",
                             "model": "claude-opus-5", "usage": _Usage()})()
    result = ApiGenerator(client=_FakeClient(resp)).generate("q", retrieved)
    assert result.error == "refusal" and result.answer == ""


def test_truncated_json_is_recorded_not_raised(retrieved):
    result = ApiGenerator(client=_FakeClient(_response('{"answer": "-9', stop_reason="max_tokens"))
                       ).generate("q", retrieved)
    assert result.error == "unparseable_json"
    assert result.stop_reason == "max_tokens"


def test_api_exception_becomes_a_record_not_a_crash(retrieved):
    """One bad call must not lose a 200-question checkpointed run."""
    class _Boom:
        def create(self, **kwargs):
            raise RuntimeError("connection reset")

    client = type("C", (), {"messages": _Boom(),
                            "beta": type("B", (), {"messages": _Boom()})()})()
    result = ApiGenerator(client=client, refusal_fallback=False).generate("q", retrieved)
    assert result.error and "connection reset" in result.error
    assert result.answer == ""


def test_request_is_evidence_bound_and_schema_constrained(retrieved):
    payload = json.dumps({"answer": "1", "evidence_numbers": [], "insufficient_evidence": True})
    client = _FakeClient(_response(payload))
    ApiGenerator(client=client, refusal_fallback=False).generate("q", retrieved)
    request = client.messages.last_request
    assert "only the evidence" in request["system"] or "only the numbered evidence" in request["system"]
    assert request["output_config"]["format"]["schema"] == ANSWER_SCHEMA
    assert request["messages"][0]["content"].startswith("Question: q")


# ---- stub generator ----------------------------------------------------------------------

# ---- local generator (the zero-cost default) ---------------------------------------------

def test_extract_json_survives_what_small_models_actually_emit():
    """A hosted API can constrain decoding to a schema; a 1.5B local model cannot. It wraps
    JSON in prose and code fences, so the parser has to dig it out rather than assume it."""
    fenced = 'Sure!\n```json\n{"answer": "-94", "evidence_numbers": [1], ' \
             '"insufficient_evidence": false}\n```'
    assert _extract_json(fenced)["answer"] == "-94"
    # a brace inside a string value must not end the object early
    assert _extract_json('{"answer": "a {b} c", "evidence_numbers": []}')["answer"] == "a {b} c"
    assert _extract_json("The answer is -94, based on rows 1 and 2.") is None
    assert _extract_json("{not valid json at all") is None


def test_extract_json_repairs_latex_style_escapes():
    """Observed in a real 250-question run: the model wrote "$ 11.6 \\% ", and `\\%` is not a
    legal JSON escape, so the whole object failed to parse and a correct answer was thrown
    away. The repair is narrow -- strip invalid escapes and retry once -- so it recovers
    formatting slips without inventing content."""
    raw = r'{"answer": "$ 11.6 \% ", "evidence_numbers": [1], "insufficient_evidence": false}'
    parsed = _extract_json(raw)
    assert parsed is not None and "11.6" in parsed["answer"]
    # a legal escape must still survive intact
    assert _extract_json(r'{"answer": "a \"quoted\" value"}')["answer"] == 'a "quoted" value'


def test_local_generator_parses_model_output(retrieved, monkeypatch):
    """Exercises generate() without loading 3GB of weights -- the model call is the one part
    that is genuinely just transformers, and the parsing around it is what breaks."""
    gen = LocalGenerator()
    monkeypatch.setattr(gen, "_run", lambda msg:
                        '```json\n{"answer": "-94", "evidence_numbers": [1, 2], '
                        '"insufficient_evidence": false}\n```')
    result = gen.generate("what was the change?", retrieved)
    assert result.answer == "-94"
    assert result.cited_chunk_ids == ["table_2", "table_3"]
    assert result.error is None


def test_local_generator_keeps_prose_when_the_model_ignores_the_format(retrieved, monkeypatch):
    """A small model that ignores the JSON instruction usually still states the answer. Keep it
    and mark it: how often this happens is a finding about the model, not a crash."""
    gen = LocalGenerator()
    monkeypatch.setattr(gen, "_run", lambda msg: "The change was -94 million.")
    result = gen.generate("q", retrieved)
    assert result.error == "unparseable_json"
    assert "-94" in result.answer


def test_executed_expression_overrides_the_models_own_arithmetic(retrieved, monkeypatch):
    """The whole point of program-of-thought: the model picks the operands, Python does the
    division. Here the model states 92 and writes an expression worth -94 -- the executed
    value wins, and the model's own answer is preserved for auditing."""
    gen = LocalGenerator(use_program=True)
    monkeypatch.setattr(gen, "_run", lambda msgs: json.dumps(
        {"answer": "92", "answer_expression": "680 - 774", "evidence_numbers": [1, 2],
         "insufficient_evidence": False}))
    result = gen.generate("what was the change?", retrieved)
    assert result.answer == "-94"
    assert result.answer_raw == "92"
    assert result.answer_source == "program"


def test_a_restated_value_does_not_override_the_answer(retrieved, monkeypatch):
    """When `answer_expression` is just the answer again it carries no extra information, and
    executing "4.35%" would rewrite a correct answer as 0.0435."""
    gen = LocalGenerator(use_program=True)
    monkeypatch.setattr(gen, "_run", lambda msgs: json.dumps(
        {"answer": "4.35%", "answer_expression": "4.35%", "evidence_numbers": [1],
         "insufficient_evidence": False}))
    result = gen.generate("what rate?", retrieved)
    assert result.answer == "4.35%"
    assert result.answer_source == "text"


def test_an_unevaluable_expression_falls_back_to_the_stated_answer(retrieved, monkeypatch):
    gen = LocalGenerator(use_program=True)
    monkeypatch.setattr(gen, "_run", lambda msgs: json.dumps(
        {"answer": "-94", "answer_expression": "revenue_2009 - revenue_2008",
         "evidence_numbers": [1], "insufficient_evidence": False}))
    result = gen.generate("q", retrieved)
    assert result.answer == "-94" and result.answer_source == "text"


def test_direct_mode_ignores_the_program_channel(retrieved, monkeypatch):
    """The `direct` ablation arm must be the original baseline, not a half-migrated one."""
    gen = LocalGenerator(use_program=False, n_shot=0)
    monkeypatch.setattr(gen, "_run", lambda msgs: json.dumps(
        {"answer": "92", "evidence_numbers": [1], "insufficient_evidence": False}))
    result = gen.generate("q", retrieved)
    assert result.answer == "92" and result.answer_source == "text"
    assert result.prompt_version == "b1-evidence-bound-v1"


def test_a_bare_expression_in_prose_is_still_salvaged(retrieved, monkeypatch):
    """Under program-of-thought a model that ignores the JSON format often emits just the
    arithmetic. It still counts as unparseable_json -- the format-compliance rate stays an
    honest measurement -- but the answer is not thrown away."""
    gen = LocalGenerator(use_program=True)
    monkeypatch.setattr(gen, "_run", lambda msgs: "680 - 774")
    result = gen.generate("q", retrieved)
    assert result.answer == "-94"
    assert result.answer_source == "program"
    assert result.error == "unparseable_json"


def test_local_generator_errors_are_recorded_not_raised(retrieved, monkeypatch):
    gen = LocalGenerator()

    def _boom(msg):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(gen, "_run", _boom)
    result = gen.generate("q", retrieved)
    assert result.error and "out of memory" in result.error


def test_default_provider_is_free(monkeypatch):
    """The zero-budget guard: leaving the config alone must never select a billed provider."""
    from generator import build_generator

    cfg = {"generation": {}}
    assert isinstance(build_generator(cfg), LocalGenerator)
    assert isinstance(build_generator(cfg, provider="stub"), StubGenerator)
    with pytest.raises(ValueError, match="unknown generation provider"):
        build_generator(cfg, provider="openai")


def test_stub_generator_runs_offline(retrieved):
    result = StubGenerator().generate("q", retrieved)
    assert result.cited_chunk_ids == ["table_2"]
    assert result.model == "stub-generator-offline"


def test_stub_generator_handles_empty_retrieval():
    result = StubGenerator().generate("q", [])
    assert result.insufficient_evidence is True and result.answer == ""
