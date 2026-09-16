"""
The B2 runner's LLM-verifier path must survive what a weaker judge does to it.

Cost three Colab attempts: the second-judge-family run (Phi-3.5) hit a TAT-QA question whose
verdict RAGAS could not parse, `verify_with_llm` raised, the run died, and -- because it died
at the same question every time -- resuming could never get past it. These tests pin the two
guards: an exception on one question is recorded and the run continues, and a torn final line
in the resume file is dropped instead of raising on every restart.

No model is loaded: the judge factory and the scorer are replaced with fakes.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "src" / "faithfulness", ROOT / "src" / "generation",
           ROOT / "src" / "grounding", ROOT / "src" / "ingestion", ROOT / "src" / "retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from staged import FaithfulnessScore, StatementSet, append_statements  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_b2", ROOT / "eval" / "baselines" / "run_b2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeScorer:
    """Raises on one qa_id's statements, scores everything else 1.0."""
    verifier_model = "fake-nli"
    verify_threshold = 0.5

    def __init__(self, poison: str):
        self.poison, self.llm, self.calls = poison, None, []

    def verify_with_llm(self, statements, contexts):
        self.calls.append(statements[0])
        if statements[0] == self.poison:
            raise RuntimeError("RagasOutputParserException: could not parse verdict")
        return FaithfulnessScore(score=1.0, n_statements=1,
                                 verdicts=[{"statement": statements[0], "supported": True}])

    def verify(self, statements, contexts):
        return self.verify_with_llm(statements, contexts)


class _FakeJudge:
    name = "local:fake-judge"


def _stage(tmp_path: Path, qa_ids):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    records = [{"qa_id": q, "question": f"q {q}", "gold_answer": "1", "failure_mode": None,
                "retrieved": [{"text": f"evidence for {q}"}],
                "generated": {"answer": f"answer {q}", "insufficient_evidence": False}}
               for q in qa_ids]
    (ckpt / "b1_rag_finqa_dev.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    append_statements(ckpt / "statements_finqa_dev.jsonl",
                      [StatementSet(qa_id=q, question=f"q {q}", answer=f"answer {q}",
                                    statements=[f"statement {q}"], error=None, judge="fake")
                       for q in qa_ids])
    return records, ckpt


def _run_verify(monkeypatch, tmp_path, records, ckpt, scorer):
    runner = _load_runner()
    monkeypatch.setattr(runner.StagedFaithfulness, "from_config",
                        classmethod(lambda cls, cfg, llm=None, **kw: scorer))
    monkeypatch.setitem(sys.modules, "generator",
                        types.SimpleNamespace(build_generator=lambda *a, **k: object()))
    monkeypatch.setitem(sys.modules, "ragas_local",
                        types.SimpleNamespace(LocalRagasLLM=lambda *a, **k: _FakeJudge()))
    args = types.SimpleNamespace(dataset="finqa", split="dev", verifier="llm", model="fake",
                                 no_4bit=True, tag="", out_tag="_T", verifier_model=None,
                                 checkpoint_every=10)
    runner._verify({}, args, records, ckpt / "statements_finqa_dev.jsonl", tmp_path)
    return json.loads((tmp_path / "b2_finqa_dev_T.json").read_text())


def test_one_judge_failure_is_recorded_not_fatal(monkeypatch, tmp_path):
    records, ckpt = _stage(tmp_path, ["a", "b", "c"])
    scorer = _FakeScorer(poison="statement b")
    report = _run_verify(monkeypatch, tmp_path, records, ckpt, scorer)

    assert scorer.calls == ["statement a", "statement b", "statement c"]   # kept going
    assert report["config"]["n_errors"] == 1
    assert report["overall"]["n_scored"] == 2 and report["overall"]["n_unscored"] == 1
    # The verifier named is the judge this run was given, not the decomposer.
    assert report["config"]["verifier"] == "local:fake-judge"

    resume = [json.loads(l) for l in open(ckpt / "b2verdicts_finqa_dev_T.jsonl")]
    failed = next(r for r in resume if r["qa_id"] == "b")
    assert failed["faithfulness"] is None and "RuntimeError" in failed["error"]


def test_resume_survives_a_torn_last_line(monkeypatch, tmp_path):
    records, ckpt = _stage(tmp_path, ["a", "b", "c"])
    good = json.dumps({"qa_id": "a", "faithfulness": 1.0, "n_statements": 1,
                       "verdicts": [{"statement": "statement a", "supported": True}]})
    (ckpt / "b2verdicts_finqa_dev_T.jsonl").write_text(good + "\n" + '{"qa_id": "b", "fai')
    scorer = _FakeScorer(poison="nothing")
    report = _run_verify(monkeypatch, tmp_path, records, ckpt, scorer)

    assert scorer.calls == ["statement b", "statement c"]      # a reused, b redone
    assert report["overall"]["n_scored"] == 3


def test_completed_run_does_not_build_a_judge(monkeypatch, tmp_path):
    records, ckpt = _stage(tmp_path, ["a"])
    (ckpt / "b2verdicts_finqa_dev_T.jsonl").write_text(json.dumps(
        {"qa_id": "a", "faithfulness": 1.0, "n_statements": 1, "verdicts": []}) + "\n")
    scorer = _FakeScorer(poison="nothing")
    monkeypatch.setitem(sys.modules, "generator", types.SimpleNamespace(
        build_generator=lambda *a, **k: pytest.fail("judge built with no work left")))
    report = _run_verify(monkeypatch, tmp_path, records, ckpt, scorer)
    assert scorer.calls == []
    assert report["config"]["verifier"] == "local:fake"        # falls back to --model
