"""
The evidence-grounding layer: which retrieved sentence supports each claim in the answer.

**This is a supporting layer, not a novelty claim.** Say so plainly -- sentence-level
answer/context matching is standard practice. What it exists for is to feed the two things
that *are* novel:

  * Contribution 1 (attribution) gets a sentence-level target. Attribution explains why a
    *chunk* was retrieved; grounding says which sentence inside it the answer actually leaned
    on, so the two halves of the explanation meet.
  * Contribution 2 (faithfulness-of-faithfulness) gets its intervention. The experiment is
    "remove the evidence that is supposedly load-bearing and check the metric reacts", and
    something has to decide what load-bearing means. `GroundingResult.support` is that
    decision, and it points at a specific `(chunk_id, sentence index)` so a perturbation can
    remove one sentence rather than a whole chunk.

Three design choices worth defending:

**Max, not mean, over evidence sentences.** A claim is grounded if *some* sentence entails it.
Averaging entailment across all retrieved sentences would punish a perfectly grounded answer
for the four irrelevant chunks that came back alongside the right one, and at top-k=5 with
~20 sentences that is the normal case, not the exception.

**Contradiction is tracked separately, not folded into the score.** An answer contradicted by
its evidence and an answer merely unsupported by it are different failures -- the first is a
hallucination against present evidence, the second is a retrieval gap. Collapsing them into
one number destroys exactly the distinction CLAUDE.md asks the faithfulness logs to preserve.

**Grounding is computed from saved B1 records, not by re-running generation.** Same principle
as `run_b1_rag.py --rescore`: the expensive half is already on disk. Grounding a full 250-
question run costs one batched NLI pass and no GPU generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from segmenter import Sentence, answer_claims, evidence_sentences

# Above this entailment probability a claim counts as supported. 0.5 is the natural decision
# boundary for a 3-way softmax and is *not* tuned against the eval set -- a threshold fitted
# to the data would make the reported grounding rate a property of the fitting, not the system.
DEFAULT_SUPPORT_THRESHOLD = 0.5
DEFAULT_CONTRADICTION_THRESHOLD = 0.5


_VALUE_RE = re.compile(r"\d[\d,]*\.?\d*")


def _sentence_values(text: str) -> list:
    """Every figure in an evidence sentence, as magnitudes."""
    values = []
    for match in _VALUE_RE.finditer(str(text)):
        try:
            values.append(abs(float(match.group().replace(",", ""))))
        except ValueError:
            continue
    return values


@dataclass
class ClaimSupport:
    """One answer claim, and the single evidence sentence that best supports it."""
    claim: str
    entailment: float
    contradiction: float
    supported: bool
    contradicted: bool
    kind: str = "text"
    sentence: str | None = None
    chunk_id: str | None = None
    sentence_index: int | None = None

    def to_json(self) -> dict:
        return {
            "claim": self.claim,
            "kind": self.kind,
            "entailment": round(self.entailment, 4),
            "contradiction": round(self.contradiction, 4),
            "supported": self.supported,
            "contradicted": self.contradicted,
            "sentence": self.sentence,
            "chunk_id": self.chunk_id,
            "sentence_index": self.sentence_index,
        }


@dataclass
class GroundingResult:
    """Grounding for one answer: per-claim support plus the aggregate."""
    support: list = field(default_factory=list)
    groundedness: float = 0.0          # mean best-entailment across claims
    supported_rate: float = 0.0        # share of claims above the support threshold
    contradicted_rate: float = 0.0
    n_claims: int = 0
    n_evidence_sentences: int = 0
    scorer: str = ""

    @property
    def cited_chunk_ids(self) -> list:
        """Chunks the grounding layer says the answer actually leaned on.

        Distinct from the model's *self-reported* citations, and the gap between the two is a
        finding in its own right: a model can cite [1] while its answer is entailed only by
        [3]. Month 7 reports both.
        """
        seen = []
        for s in self.support:
            if s.supported and s.chunk_id and s.chunk_id not in seen:
                seen.append(s.chunk_id)
        return seen

    def to_json(self) -> dict:
        return {
            "groundedness": round(self.groundedness, 4),
            "supported_rate": round(self.supported_rate, 4),
            "contradicted_rate": round(self.contradicted_rate, 4),
            "n_claims": self.n_claims,
            "n_evidence_sentences": self.n_evidence_sentences,
            "grounded_chunk_ids": self.cited_chunk_ids,
            "scorer": self.scorer,
            "support": [s.to_json() for s in self.support],
        }


class Grounder:
    """Pairs every answer claim against every evidence sentence and keeps the best match."""

    def __init__(self, scorer, support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
                 contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD):
        self.scorer = scorer
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold

    @classmethod
    def from_config(cls, cfg: dict | None = None, backend: str | None = None):
        from config_utils import get as cfg_get, load_config

        from entailment import build_scorer

        cfg = cfg or load_config()
        return cls(
            scorer=build_scorer(cfg, backend=backend),
            support_threshold=float(cfg_get(cfg, "grounding.support_threshold",
                                            DEFAULT_SUPPORT_THRESHOLD)),
            contradiction_threshold=float(cfg_get(cfg, "grounding.contradiction_threshold",
                                                  DEFAULT_CONTRADICTION_THRESHOLD)),
        )

    def ground(self, answer: str, retrieved, question: str | None = None,
               expression: str | None = None) -> GroundingResult:
        """Ground one answer against its retrieved evidence."""
        return self.ground_batch([(answer, retrieved, question, expression)])[0]

    def ground_batch(self, items) -> list:
        """Ground several answers in one pass.

        Items are `(answer, retrieved, question, expression)`; a 3-tuple is accepted for
        callers with no program-of-thought expression to offer.

        Only *text* claims reach the NLI model, and every text pair across the whole batch is
        collected before anything is scored, so the batch is one set of forward passes.
        Numeric claims never touch the model at all -- provenance is a lookup, which is both
        exact and free. On FinQA, where almost every answer is arithmetic, that removes the
        NLI model from the critical path entirely.
        """
        pairs: list[tuple[str, str]] = []
        layout = []
        for item in items:
            answer, retrieved, question = item[0], item[1], item[2]
            expression = item[3] if len(item) > 3 else None
            claims = answer_claims(answer, question, expression)
            sentences = evidence_sentences(retrieved)
            text_claims = [c for c in claims if c.kind == "text"]
            layout.append((claims, sentences, len(pairs), text_claims))
            for claim in text_claims:
                for sentence in sentences:
                    pairs.append((sentence.text, claim.text))

        scores = self.scorer.score_pairs(pairs) if pairs else []
        name = getattr(self.scorer, "model_name", type(self.scorer).__name__)

        return [self._assemble(claims, sentences, scores, offset, text_claims, name)
                for claims, sentences, offset, text_claims in layout]

    def _assemble(self, claims, sentences, scores, offset, text_claims,
                  scorer_name) -> GroundingResult:
        support = []
        n_sent = len(sentences)
        text_position = {id(c): i for i, c in enumerate(text_claims)}

        for claim in claims:
            if claim.kind == "numeric":
                support.append(self._ground_numeric(claim, sentences))
                continue
            c_i = text_position[id(claim)]
            best: tuple[float, Sentence | None] = (0.0, None)
            worst_contradiction = 0.0
            for s_i, sentence in enumerate(sentences):
                result = scores[offset + c_i * n_sent + s_i]
                worst_contradiction = max(worst_contradiction, result.contradiction)
                if result.entailment > best[0]:
                    best = (result.entailment, sentence)
            entailment, sentence = best
            supported = entailment >= self.support_threshold
            support.append(ClaimSupport(
                claim=claim.text, kind="text",
                entailment=entailment, contradiction=worst_contradiction,
                supported=supported,
                # Contradiction only counts for a claim that found no support. Retrieved
                # evidence contains many facts, and an NLI model reads "revenue 2018 was
                # 440.7" as contradicting a claim about 2019 -- so the raw maximum fires on
                # ordinary multi-year tables and says nothing about faithfulness. Measured:
                # gating this way took TAT-QA's contradicted rate from 0.440 to a figure that
                # actually means "asserted something the evidence denies". The raw maximum is
                # still recorded in `contradiction` for anyone who wants it.
                contradicted=(not supported
                              and worst_contradiction >= self.contradiction_threshold),
                sentence=sentence.text if sentence else None,
                chunk_id=sentence.chunk_id if sentence else None,
                sentence_index=sentence.index if sentence else None,
            ))

        n = len(support) or 1
        return GroundingResult(
            support=support,
            groundedness=sum(s.entailment for s in support) / n,
            supported_rate=sum(s.supported for s in support) / n,
            contradicted_rate=sum(s.contradicted for s in support) / n,
            n_claims=len(support),
            n_evidence_sentences=n_sent,
            scorer=scorer_name,
        )

    @staticmethod
    def _ground_numeric(claim, sentences) -> ClaimSupport:
        """Locate the evidence sentence that supplies a numeric operand.

        Exact on magnitude, with a relative tolerance for the rounding the documents
        themselves do. Magnitude rather than signed value because a table renders negatives as
        "(774)" and a model writes "-774" -- and because in `680-774` the minus is an
        operator, not a sign.

        A found operand scores 1.0 and a missing one 0.0: provenance is a fact, not a
        probability, and blurring it would make the numeric and text scales incomparable in
        the aggregate.
        """
        target = claim.value
        for sentence in sentences:
            for value in _sentence_values(sentence.text):
                if abs(value - target) <= max(1e-6, 1e-4 * max(abs(value), abs(target))):
                    return ClaimSupport(
                        claim=claim.text, kind="numeric",
                        entailment=1.0, contradiction=0.0,
                        supported=True, contradicted=False,
                        sentence=sentence.text, chunk_id=sentence.chunk_id,
                        sentence_index=sentence.index,
                    )
        return ClaimSupport(claim=claim.text, kind="numeric", entailment=0.0,
                            contradiction=0.0, supported=False, contradicted=False)
