"""
Faithfulness split into a cached expensive stage and a cheap repeatable one.

RAGAS computes faithfulness in two stages, and the whole feasibility of Contribution 2 turns
on the fact that they depend on different inputs:

    stage 1  decompose   (question, answer)          --LLM-->   atomic statements
    stage 2  verify      (statements, context)       --NLI-->   supported / not

Contribution 2 perturbs **the context** and asks whether the faithfulness score moves in the
direction it should. The question and answer are held fixed by construction -- that is what
makes it a controlled experiment. So stage 1's output is *invariant across every perturbation
of a given question*, while only stage 2 has to re-run.

Calling `ragas.evaluate()` once per perturbation would redo stage 1 every time. At 250
questions x 2 datasets x several conditions x 3 repeats that is thousands of LLM calls and
close to a day of free-tier GPU time -- and every one of them recomputing a result that
cannot have changed. Splitting the stages turns the perturbation loop into local NLI: no LLM,
no GPU, no cost, fast enough to run many conditions and the repeats CLAUDE.md asks for.

This is the same guardrail the attribution engine follows ("never re-run the full pipeline per
perturbation; pre-compute the invariant part once"), applied to the faithfulness half.

**What is and is not ours.** The prompts, the decomposition and the verifier are RAGAS's --
that matters, because Contribution 2's claim is about *RAGAS's* metric, and a reimplementation
would be testing something else. This module only decides *when* each stage runs and caches
the part that cannot change. `verify()` calls the same `nli_classifier` instance
`FaithfulnesswithHHEM` uses, and `_compute_score` is RAGAS's own formula.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / "generation"), str(_SRC / "faithfulness")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ragas_local import no_paid_providers  # noqa: E402


@dataclass
class StatementSet:
    """Stage 1's output for one question: the atomic claims RAGAS read out of the answer.

    Cached to disk because it is the expensive half and it cannot change under a context
    perturbation. `qa_id` keys the cache; `answer` is stored alongside so a cache entry can be
    invalidated when the answer it was derived from changes -- silently reusing statements
    from a different answer would corrupt every downstream score without failing.
    """
    qa_id: str
    question: str
    answer: str
    statements: list = field(default_factory=list)
    error: str | None = None
    # Which model produced this decomposition. A faithfulness number is only interpretable
    # alongside the judge that made it, and the verify phase runs in a separate process (often
    # on a different machine) with no way to know -- so it travels with the cache entry.
    judge: str | None = None

    def to_json(self) -> dict:
        return {"qa_id": self.qa_id, "question": self.question, "answer": self.answer,
                "statements": self.statements, "error": self.error, "judge": self.judge}

    @classmethod
    def from_json(cls, data: dict) -> "StatementSet":
        return cls(qa_id=data["qa_id"], question=data.get("question", ""),
                   answer=data.get("answer", ""), statements=data.get("statements", []),
                   error=data.get("error"), judge=data.get("judge"))


@dataclass
class FaithfulnessScore:
    """Stage 2's output: RAGAS's faithfulness score plus the per-statement verdicts.

    The verdicts are kept, not just the mean. Contribution 2 needs to know *which* statement
    stopped being supported when a piece of evidence was removed -- a score that moved from
    0.75 to 0.50 without saying which claim lost its support is not evidence of anything.
    """
    score: float
    verdicts: list = field(default_factory=list)
    n_statements: int = 0

    def to_json(self) -> dict:
        return {"faithfulness": None if self.score != self.score else round(self.score, 4),
                "verdicts": self.verdicts, "n_statements": self.n_statements}


# The verifier for stage 2.
#
# RAGAS's own `FaithfulnesswithHHEM` uses Vectara's hallucination-evaluation model, and that
# was the first choice here -- purpose-built, and from a lineage unrelated to anything else in
# the pipeline. It does not work on this project's stack: HHEM ships custom remote code that
# predates `transformers` 5.x and dies with
#     AttributeError: 'HHEMv2ForSequenceClassification' object has no attribute
#                     'all_tied_weights_keys'
# Downgrading transformers is not available as a fix -- 5.15.0 produced every B1 number, and
# CLAUDE.md pins it for exactly that reason.
#
# So stage 2 runs a standard NLI cross-encoder instead. This is the same substitution RAGAS
# sanctions (FaithfulnesswithHHEM exists precisely to replace the LLM verifier with a model);
# only the model differs. Two properties are preserved, and they are the ones that matter:
#
#   * The generator never grades its own answers -- verification is a separate model family
#     from Qwen.
#   * The verifier is a *different checkpoint* from the grounding layer's
#     (`grounding.nli_model`), so Contribution 2 is not testing one model against itself.
#     The independence is strongest exactly where it matters most: on FinQA, 93% of grounding
#     decisions are operand provenance -- a deterministic lookup that uses no model at all --
#     so there is no shared model to be circular about.
DEFAULT_VERIFIER_MODEL = "cross-encoder/nli-deberta-v3-base"

# A statement counts as supported at the natural decision boundary of the softmax. Not tuned:
# fitting this to the eval set would make the reported faithfulness a property of the fitting.
DEFAULT_VERIFY_THRESHOLD = 0.5


class StagedFaithfulness:
    """RAGAS faithfulness with stage 1 cached and stage 2 repeatable.

    Construct once, decompose over the dataset (expensive, checkpointed), then call `verify()`
    as many times as the experiment needs (cheap, local).
    """

    def __init__(self, llm=None, device: str | None = None, batch_size: int = 16,
                 verifier_model: str = DEFAULT_VERIFIER_MODEL,
                 verify_threshold: float = DEFAULT_VERIFY_THRESHOLD):
        self.llm = llm
        self.device = device
        self.batch_size = batch_size
        self.verifier_model = verifier_model
        self.verify_threshold = verify_threshold
        self._metric = None
        self._verifier = None

    @classmethod
    def from_config(cls, cfg=None, llm=None, **overrides):
        from config_utils import get as cfg_get, load_config

        cfg = cfg or load_config()
        kwargs = dict(
            llm=llm,
            device=cfg_get(cfg, "faithfulness.device"),
            batch_size=int(cfg_get(cfg, "faithfulness.batch_size", 16)),
            verifier_model=cfg_get(cfg, "faithfulness.verifier_model",
                                   DEFAULT_VERIFIER_MODEL),
            verify_threshold=float(cfg_get(cfg, "faithfulness.verify_threshold",
                                           DEFAULT_VERIFY_THRESHOLD)),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def verifier(self):
        """Stage 2's model. Loaded lazily and separately from the judge, so a verify-only run
        never pays for the generator."""
        if self._verifier is None:
            sys.path.insert(0, str(_SRC / "grounding"))
            from entailment import EntailmentScorer

            self._verifier = EntailmentScorer(model_name=self.verifier_model,
                                              batch_size=self.batch_size,
                                              device=self.device)
        return self._verifier

    @property
    def metric(self):
        """RAGAS's Faithfulness, constructed under the no-billing guard, for its *prompts*.

        Only stage 1 uses this. The decomposition prompt has to be RAGAS's own, because
        Contribution 2's claim is about RAGAS's metric and a reimplementation would be testing
        something else. `use_hhem=False` because this instance is never asked to verify.
        """
        if self._metric is None:
            from ragas_local import build_faithfulness_metric

            with no_paid_providers():
                self._metric = build_faithfulness_metric(self.llm, use_hhem=False)
        return self._metric

    # -- stage 1: expensive, cacheable ------------------------------------------------------
    def decompose(self, qa_id: str, question: str, answer: str) -> StatementSet:
        """Ask the judge LLM to break one answer into atomic statements.

        Depends only on (question, answer) -- never on the retrieved context. That is the
        property the whole caching scheme rests on, and it is RAGAS's design, not an
        assumption: `Faithfulness._create_statements` reads `row["response"]` and
        `row["user_input"]` and nothing else.
        """
        judge = getattr(self.llm, "name", None)
        if not str(answer or "").strip():
            # A declined or empty answer has no statements. RAGAS returns NaN for this case;
            # recording it as an empty set keeps that distinguishable from a failed call.
            return StatementSet(qa_id=qa_id, question=question, answer=answer,
                                statements=[], judge=judge)
        try:
            import asyncio

            row = {"user_input": question, "response": answer, "retrieved_contexts": []}
            with no_paid_providers():
                output = asyncio.run(self.metric._create_statements(row, None))
            return StatementSet(qa_id=qa_id, question=question, answer=answer,
                                statements=list(output.statements), judge=judge)
        except Exception as exc:                      # one bad question must not kill the run
            return StatementSet(qa_id=qa_id, question=question, answer=answer,
                                statements=[], error=f"{type(exc).__name__}: {exc}",
                                judge=judge)

    # -- stage 2: cheap, repeatable ---------------------------------------------------------
    def verify(self, statements: list, contexts: list) -> FaithfulnessScore:
        """Score cached statements against a (possibly perturbed) context. Local, free.

        This is the function Month 6 calls in a loop. It touches no LLM and no network, so a
        perturbation sweep costs the same as reading the files.
        """
        statements = [s for s in statements if str(s).strip()]
        if not statements:
            return FaithfulnessScore(score=float("nan"), verdicts=[], n_statements=0)

        # One premise per statement: the whole retrieved context, exactly as RAGAS joins it.
        premise = "\n".join(str(c) for c in contexts)
        results = self.verifier.score_pairs([(premise, s) for s in statements])

        verdicts = [
            {"statement": s,
             "supported": r.entailment >= self.verify_threshold,
             "entailment": round(r.entailment, 4)}
            for s, r in zip(statements, results)
        ]
        # RAGAS's own formula: the share of statements the verifier accepts.
        supported = sum(v["supported"] for v in verdicts)
        return FaithfulnessScore(score=supported / len(verdicts), verdicts=verdicts,
                                 n_statements=len(statements))

    def verify_with_llm(self, statements: list, contexts: list) -> FaithfulnessScore:
        """RAGAS's *unmodified* verifier: the judge LLM decides each statement.

        This is the reference the NLI verifier is checked against, and it exists because
        substituting a verifier is a deviation that has to be defended rather than asserted.
        RAGAS's own `Faithfulness` prompts the LLM with (context, statements) and reads back a
        0/1 verdict plus a reason; that path is used verbatim here.

        Not used for reported B2 numbers. It costs one LLM call per question *per context*,
        which is exactly the property that makes it unusable inside Month 6's perturbation
        loop -- and exactly why the NLI verifier exists. It is affordable on a subset, which
        is all a cross-check needs.
        """
        statements = [s for s in statements if str(s).strip()]
        if not statements:
            return FaithfulnessScore(score=float("nan"), verdicts=[], n_statements=0)

        import asyncio

        row = {"retrieved_contexts": [str(c) for c in contexts]}
        with no_paid_providers():
            output = asyncio.run(self.metric._create_verdicts(row, statements, None))

        verdicts = [{"statement": a.statement, "supported": bool(a.verdict),
                     "reason": getattr(a, "reason", None)}
                    for a in output.statements]
        if not verdicts:
            return FaithfulnessScore(score=float("nan"), verdicts=[], n_statements=0)
        supported = sum(v["supported"] for v in verdicts)
        return FaithfulnessScore(score=supported / len(verdicts), verdicts=verdicts,
                                 n_statements=len(verdicts))


# -- statement cache ------------------------------------------------------------------------

def load_statements(path) -> dict:
    """Read a statement cache into {qa_id: StatementSet}. Missing file -> empty cache."""
    path = Path(path)
    if not path.exists():
        return {}
    cache = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = StatementSet.from_json(json.loads(line))
            except (json.JSONDecodeError, KeyError):
                # A run killed mid-write leaves one torn line; drop it rather than refusing
                # to resume, exactly as the B1 checkpoint loader does.
                break
            cache[entry.qa_id] = entry
    return cache


def append_statements(path, entries) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_json()) + "\n")


def cache_is_stale(entry: StatementSet, answer: str) -> bool:
    """True when a cached decomposition was derived from a different answer.

    Guards the one way this cache can corrupt results silently: B1 is re-run, answers change,
    and stale statements get verified against the new contexts. Cheap to check, and the
    failure it prevents produces plausible-looking numbers rather than an error.
    """
    return str(entry.answer or "").strip() != str(answer or "").strip()
