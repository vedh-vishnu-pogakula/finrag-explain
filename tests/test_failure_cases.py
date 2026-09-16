"""
Tests for the Month 6 deliverable scripts: the failure-case labeler and the B3 table.

Both scripts are joins over result files with a few pure decision functions in the middle.
The decision functions are what a reviewer will question ("why is this row a false
positive?"), so those are pinned here; the file plumbing is exercised end to end by running
the scripts on the checked-in results.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = ROOT / "eval" / "baselines" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fc = _load("export_failure_cases")
b3 = _load("build_b3_table")


# ---- label_case -------------------------------------------------------------------------

def _label(**kw):
    base = dict(faithfulness=None, correct=False, failure_mode=None, grounded=None,
                scores=None, threshold=0.5)
    base.update(kw)
    return fc.label_case(**base)


def test_no_statements_is_unscorable_and_nothing_else():
    assert _label(faithfulness=None, correct=True, grounded=True) == ["unscorable"]


def test_correct_grounded_but_unfaithful_is_false_negative():
    # The arithmetic-vs-entailment mechanism: 3280/19686*100 = 16.66, both operands in the
    # evidence, NLI scores 0.0.
    assert _label(faithfulness=0.0, correct=True, grounded=True) == ["false_negative"]


def test_correct_unfaithful_without_provenance_is_not_a_false_negative():
    # No numeric claims (grounded=None): we cannot prove the answer is supported, so we
    # do not blame the verifier.
    assert _label(faithfulness=0.0, correct=True, grounded=None) == []


def test_wrong_with_gold_retrieved_but_faithful_is_false_positive():
    assert _label(faithfulness=1.0, correct=False, failure_mode="generation") == ["false_positive"]


def test_wrong_because_retrieval_failed_is_not_the_verifiers_fault():
    # Evidence genuinely doesn't support the answer; "faithful" is wrong but the verifier
    # only saw what retrieval gave it. Only an ungrounded operand would make it a FP.
    assert _label(faithfulness=1.0, correct=False, failure_mode="retrieval") == []
    assert _label(faithfulness=1.0, correct=False, failure_mode="retrieval",
                  grounded=False) == ["false_positive"]


def test_missing_operand_scored_faithful_is_false_positive_even_if_answer_correct():
    assert _label(faithfulness=1.0, correct=True, grounded=False) == ["false_positive"]


def test_threshold_is_respected():
    assert _label(faithfulness=0.49, correct=True, grounded=True) == ["false_negative"]
    assert _label(faithfulness=0.5, correct=True, grounded=True) == []


def test_targeted_removal_that_does_not_move_the_score_is_insensitive():
    scores = {"control": 1.0, "targeted": 1.0, "random": 0.5}
    assert _label(faithfulness=1.0, correct=True, grounded=True, scores=scores) == ["insensitive"]


def test_targeted_drop_no_larger_than_random_drop_is_nonspecific():
    scores = {"control": 1.0, "targeted": 0.5, "random": 0.5}
    assert _label(faithfulness=1.0, correct=True, grounded=True, scores=scores) == ["nonspecific"]
    scores = {"control": 1.0, "targeted": 0.5, "random": 0.0}
    assert "nonspecific" in _label(faithfulness=1.0, correct=True, grounded=True, scores=scores)


def test_specific_response_earns_no_perturbation_label():
    scores = {"control": 1.0, "targeted": 0.0, "random": 0.5}
    assert _label(faithfulness=1.0, correct=True, grounded=True, scores=scores) == []


def test_zero_control_score_cannot_be_insensitive():
    # Nothing to lower: a 0.0 that stays 0.0 says nothing about sensitivity.
    scores = {"control": 0.0, "targeted": 0.0, "random": 0.0}
    assert _label(faithfulness=0.0, correct=False, failure_mode="generation",
                  scores=scores) == []


def test_partial_perturbation_record_is_ignored():
    scores = {"control": 1.0, "targeted": None, "random": 0.5}
    assert _label(faithfulness=1.0, correct=True, grounded=True, scores=scores) == []


def test_labels_can_stack():
    scores = {"control": 1.0, "targeted": 1.0, "random": 1.0}
    got = _label(faithfulness=1.0, correct=False, failure_mode="generation", scores=scores)
    assert got == ["false_positive", "insensitive"]


# ---- restates_figure --------------------------------------------------------------------

def test_restatement_matches_numerically_not_textually():
    ev = ["the contingent rental of 2009 is 19", "total revenue was 1,400 million"]
    assert fc.restates_figure("19", ev) is True
    assert fc.restates_figure("-19.0", ev) is True          # sign and precision ignored
    assert fc.restates_figure("1400", ev) is True           # thousands separator
    assert fc.restates_figure("14", ev) is False            # computed, not copied
    assert fc.restates_figure("$19", ev) is True


def test_restatement_is_undefined_for_text_answers():
    assert fc.restates_figure("the total number of days", ["anything"]) is None
    assert fc.restates_figure(None, ["anything"]) is None


# ---- operands_grounded ------------------------------------------------------------------

def _grounding(kinds_supported):
    return {"grounding": {"support": [{"kind": k, "supported": s} for k, s in kinds_supported],
                          "groundedness": 0.0}}


def test_operands_grounded_uses_numeric_claims_only():
    assert fc.operands_grounded(_grounding([("numeric", True), ("numeric", True)])) is True
    assert fc.operands_grounded(_grounding([("numeric", True), ("numeric", False)])) is False
    assert fc.operands_grounded(_grounding([("text", True)])) is None
    assert fc.operands_grounded(None) is None


# ---- B3 table ----------------------------------------------------------------------------

def test_is_grounded_falls_back_to_nli_groundedness_for_text_answers():
    rec = {"grounding": {"support": [{"kind": "text", "supported": True}], "groundedness": 0.75}}
    assert b3.is_grounded(rec, 0.5) is True
    assert b3.is_grounded(rec, 0.8) is False
    rec = {"grounding": {"support": [{"kind": "numeric", "supported": False}], "groundedness": 1.0}}
    assert b3.is_grounded(rec, 0.5) is False    # operands override the NLI number
    assert b3.is_grounded(None, 0.5) is False


def test_selective_accuracy_splits_by_flag():
    flags = [(True, True), (True, False), (False, False), (False, False), (True, True)]
    s = b3.selective_accuracy(flags)
    assert s["n"] == 5
    assert s["coverage"] == 0.6
    assert abs(s["accuracy_verified"] - 2 / 3) < 1e-4      # rounded to 4 dp in the table
    assert s["accuracy_unverified"] == 0.0
    assert s["n_correct_verified"] == 2 and s["n_correct_total"] == 2


def test_selective_accuracy_with_nothing_verified_reports_none_not_zero():
    s = b3.selective_accuracy([(False, True), (False, False)])
    assert s["coverage"] == 0.0 and s["accuracy_verified"] is None
