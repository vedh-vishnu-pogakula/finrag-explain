"""
Evidence perturbation for Contribution 2: remove what the answer depends on, and check whether
the faithfulness metric notices.

The experiment is a controlled comparison, and the control is the whole point. Removing *any*
sentence from the retrieved context lowers a faithfulness score somewhat -- there is simply less
text for the verifier to match against. A metric that drops when load-bearing evidence is
removed has therefore demonstrated nothing on its own. The claim rests on the **difference**
between removing the evidence the answer actually used and removing an equal amount of evidence
it did not:

    control      the retrieved context, unchanged
    targeted     the sentence(s) the grounding layer says supplied the answer, removed
    random       an equal number of *other* sentences, removed

A metric that tracks grounding shows `targeted` well below `random`. A metric that tracks
surface overlap shows them roughly equal -- it lost the same amount of text either way.

**Where the target comes from matters.** For numeric answers the grounding layer identifies the
supplying sentence by operand provenance: a deterministic lookup of whether that figure occurs
in that sentence. No model is involved, so the perturbation target is not something a verifier
could be accused of having chosen for itself. That independence is what makes the comparison
meaningful rather than circular.

**Determinism.** Random removal is seeded per question, so the whole sweep reproduces exactly.
An unseeded control arm would make a null result impossible to distinguish from noise.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

CONTROL = "control"
TARGETED = "targeted"
RANDOM = "random"


@dataclass
class PerturbedContext:
    """One experimental condition: the surviving context plus what was taken out of it."""
    condition: str
    contexts: list = field(default_factory=list)
    removed: list = field(default_factory=list)          # (chunk_id, sentence_index, text)
    n_removed: int = 0
    n_sentences: int = 0

    def to_json(self) -> dict:
        return {"condition": self.condition, "n_removed": self.n_removed,
                "n_sentences": self.n_sentences,
                "removed": [{"chunk_id": c, "sentence_index": i, "text": t}
                            for c, i, t in self.removed]}


def _rebuild(sentences, drop: set) -> list:
    """Reassemble the context from surviving sentences, preserving chunk order.

    Sentences are rejoined per chunk rather than emitted individually, because RAGAS's verifier
    receives the context as one block of text and a chunk split into fragments would change the
    input distribution independently of what was removed -- confounding the very comparison the
    experiment is making.
    """
    by_chunk: dict = {}
    for pos, sentence in enumerate(sentences):
        if pos in drop:
            continue
        by_chunk.setdefault(sentence.chunk_id, []).append(sentence.text)
    return [" ".join(parts) for parts in by_chunk.values() if parts]


def build_conditions(sentences, support, seed: int = 0) -> list:
    """Produce the three conditions for one question.

    `sentences` is `segmenter.evidence_sentences(retrieved)`; `support` is the
    `GroundingResult.support` list. Returns [] when the answer has no supported claim -- with
    nothing identified as load-bearing there is no targeted arm, and reporting a control-only
    result as if it were an experiment would be misleading.
    """
    index = {(s.chunk_id, s.index): pos for pos, s in enumerate(sentences)}
    targets = set()
    for claim in support:
        if not claim.supported or claim.chunk_id is None:
            continue
        pos = index.get((claim.chunk_id, claim.sentence_index))
        if pos is not None:
            targets.add(pos)
    if not targets or len(targets) >= len(sentences):
        return []

    # Sample the random arm from sentences that are NOT load-bearing, and take exactly as many
    # as the targeted arm removes. Equal volume is what isolates "which evidence" from "how
    # much evidence" -- without it the two arms differ on two variables at once.
    pool = [p for p in range(len(sentences)) if p not in targets]
    rng = random.Random(seed)
    decoys = set(rng.sample(pool, min(len(targets), len(pool))))

    def make(condition, drop):
        return PerturbedContext(
            condition=condition,
            contexts=_rebuild(sentences, drop),
            removed=[(sentences[p].chunk_id, sentences[p].index, sentences[p].text)
                     for p in sorted(drop)],
            n_removed=len(drop),
            n_sentences=len(sentences),
        )

    return [make(CONTROL, set()), make(TARGETED, targets), make(RANDOM, decoys)]
