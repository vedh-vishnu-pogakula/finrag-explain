"""
Top-level entry point for Contribution 1: question in, per-chunk attribution out.

    explanation = explain_query(retriever, "what was the change in net revenue?",
                                doc_id="V/2008/page_17.pdf-1", method="shapley")

One property worth understanding, because it is where most of the compute savings actually
come from: **explaining k chunks costs the same as explaining one.** The perturbation
coalitions depend only on the query, not on which chunk is being explained, and
`PerturbationScorer` scores every chunk of the document in the same matmul. So the second
through k-th targets are pure column reads out of the cache -- zero additional embeddings.
That is why the default explains the whole retrieved top-k rather than just the top hit.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / "attribution"), str(_SRC / "retrieval"),
           str(_SRC / "ingestion")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from attributors import build_attributor  # noqa: E402
from perturbation import PerturbationScorer, ranking_value_fn, score_value_fn  # noqa: E402
from segmentation import QueryUnit, segment_query  # noqa: E402


@dataclass
class UnitAttribution:
    index: int
    text: str
    kind: str
    weight: float

    def to_json(self) -> dict:
        return {"index": self.index, "text": self.text, "kind": self.kind,
                "weight": round(float(self.weight), 6)}


@dataclass
class ChunkExplanation:
    """Why one retrieved chunk scored what it did."""
    chunk_id: str
    chunk_type: str
    rank: int
    base_score: float
    attributions: list = field(default_factory=list)

    def top_units(self, n: int = 3) -> list:
        return sorted(self.attributions, key=lambda a: -a.weight)[:n]

    def to_json(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "rank": self.rank,
            "base_score": round(float(self.base_score), 6),
            "attributions": [a.to_json() for a in self.attributions],
        }


@dataclass
class QueryExplanation:
    question: str
    doc_id: str | None
    method: str
    mode: str
    units: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    ranking_attributions: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "question": self.question,
            "doc_id": self.doc_id,
            "method": self.method,
            "mode": self.mode,
            "units": [u.to_json() for u in self.units],
            "chunks": [c.to_json() for c in self.chunks],
            "ranking_attributions": [a.to_json() for a in self.ranking_attributions],
            "stats": self.stats,
        }

    def summary(self, n_units: int = 3) -> str:
        """One readable line per explained chunk -- what the demo and the paper's qualitative
        examples are built from."""
        lines = [f'Q: "{self.question}"  [{self.method}/{self.mode}]']
        for chunk in self.chunks:
            drivers = ", ".join(f'"{a.text}" {a.weight:+.4f}' for a in chunk.top_units(n_units))
            lines.append(f"  #{chunk.rank + 1} {chunk.chunk_id} "
                         f"(score {chunk.base_score:.4f}) <- {drivers}")
        return "\n".join(lines)


def explain_query(retriever, question: str, doc_id: str | None = None, method: str = "shapley",
                  mode: str = "score", top_k: int = 3, use_ner: bool = True,
                  rbo_depth: int = 10, **attributor_kwargs) -> QueryExplanation:
    """Attribute a query's retrieval behaviour to its individual units.

    mode="score"   -- explain each of the top-k retrieved chunks' similarity scores.
    mode="ranking" -- explain the top-k ordering as a whole (RankingSHAP's formulation).
    """
    units: list[QueryUnit] = segment_query(question, use_ner=use_ner)
    scorer = PerturbationScorer(retriever, question, units, doc_id)
    attributor = build_attributor(method, **attributor_kwargs)

    base = scorer.base_scores()
    order = np.argsort(-base)[:top_k]

    explanation = QueryExplanation(question=question, doc_id=doc_id, method=method, mode=mode,
                                   units=units)

    if not units:
        explanation.stats = scorer.stats()
        return explanation

    if mode == "ranking":
        result = attributor.attribute(scorer, ranking_value_fn(base, depth=rbo_depth))
        explanation.ranking_attributions = _to_units(units, result.weights)
        explanation.stats = {**scorer.stats(), **result.meta}
        return explanation

    if mode != "score":
        raise ValueError(f"mode must be 'score' or 'ranking', got {mode!r}")

    meta: dict = {}
    for rank, col in enumerate(order):
        result = attributor.attribute(scorer, score_value_fn(int(col)))
        meta = result.meta
        chunk = scorer.candidates[int(col)]
        explanation.chunks.append(ChunkExplanation(
            chunk_id=chunk.chunk_id,
            chunk_type=chunk.chunk_type,
            rank=rank,
            base_score=float(base[col]),
            attributions=_to_units(units, result.weights),
        ))
    explanation.stats = {**scorer.stats(), **meta}
    return explanation


def _to_units(units: list[QueryUnit], weights: np.ndarray) -> list:
    return [UnitAttribution(index=u.index, text=u.text, kind=u.kind, weight=float(w))
            for u, w in zip(units, weights)]
