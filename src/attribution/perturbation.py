"""
The perturbation engine: scoring masked variants of a query, cheaply.

Everything in Contribution 1 reduces to one operation -- "score this query with these units
removed" -- executed thousands of times per question. The entire feasibility of the month
rests on that operation being a cached lookup into a batched matrix multiply rather than a
retrieval call.

Three properties make it affordable, and they are worth being able to state precisely:

1. **Chunk embeddings are computed once**, at index build time, and never recomputed. A
   perturbation changes the query, never the corpus.
2. **All variants are embedded in one batch and scored in one matmul**, via
   `Retriever.score_query_variants`. The estimators below deliberately collect every coalition
   they will ever need *before* scoring any of them, so a Shapley run is one batch, not one
   batch per permutation.
3. **Identical coalitions are scored once.** Monte Carlo permutation sampling revisits the
   same subsets constantly -- the empty and full coalitions appear in every permutation -- and
   `n_embedded` vs `n_requested` on the result records how much that saved.

The generator is never involved. `PerturbationScorer` holds a retriever and nothing else, so
the anti-pattern CLAUDE.md forbids -- calling an LLM inside a perturbation loop -- is not
reachable from this module.

Two value functions are provided, because "what drove the score" and "what drove the ranking"
are different questions:

  * `score_value_fn`  -- explains one chunk's similarity score. Per-chunk, directly
    interpretable, and what you want when asking why *this* row was retrieved.
  * `ranking_value_fn` -- explains the whole top-k ordering, using rank-biased overlap against
    the unperturbed ranking. This is the RankingSHAP formulation: the value of a coalition is
    how much of the original ranking it reproduces.
"""
from __future__ import annotations

import numpy as np

from segmentation import QueryUnit, render


def rbo(ranking_a: list, ranking_b: list, p: float = 0.9, depth: int | None = None) -> float:
    """Rank-biased overlap, normalized so identical rankings score exactly 1.0.

    RBO is used rather than Kendall's tau because retrieval explanations care far more about
    the top of the list than the tail: a perturbation that reorders ranks 1 and 2 matters, one
    that reorders ranks 40 and 41 does not. The `p` parameter sets how sharply that weighting
    decays -- 0.9 puts roughly 86% of the weight in the top 10.

    The raw truncated RBO of two identical lists is (1 - p^depth), not 1, so it is divided
    through by that constant. Without the normalization a Shapley run over this value function
    would report that the full query only achieves 0.65 of "its own" ranking, which is an
    artifact of the measure rather than anything about the query.
    """
    depth = depth or min(len(ranking_a), len(ranking_b))
    if depth == 0:
        return 0.0
    seen_a, seen_b, total = set(), set(), 0.0
    for d in range(1, depth + 1):
        if d <= len(ranking_a):
            seen_a.add(ranking_a[d - 1])
        if d <= len(ranking_b):
            seen_b.add(ranking_b[d - 1])
        total += (p ** (d - 1)) * len(seen_a & seen_b) / d
    return float((1 - p) * total / (1 - p ** depth))


class PerturbationScorer:
    """Scores masked variants of one query against one document's candidate chunks.

    A mask is a boolean vector over the query's units: True = keep, False = blank out.
    """

    def __init__(self, retriever, question: str, units: list[QueryUnit], doc_id: str | None):
        self.retriever = retriever
        self.question = question
        self.units = units
        self.doc_id = doc_id
        self.n_units = len(units)
        self._cache: dict[bytes, np.ndarray] = {}
        self.candidates: list = []
        self.n_requested = 0
        self.n_embedded = 0

    # ---- masks ---------------------------------------------------------------------------

    def full_mask(self) -> np.ndarray:
        return np.ones(self.n_units, dtype=bool)

    def empty_mask(self) -> np.ndarray:
        """The all-removed baseline. Rendering it leaves an empty query, which the embedder
        still maps to a real vector (the BGE instruction prefix alone). That vector is the
        reference point every attribution is measured against, and it is a legitimate one:
        it is literally "this retriever, asked nothing"."""
        return np.zeros(self.n_units, dtype=bool)

    # ---- scoring -------------------------------------------------------------------------

    def score(self, masks: np.ndarray) -> np.ndarray:
        """Score a batch of masks. Returns (n_masks, n_candidate_chunks).

        Deduplicates against everything scored so far, embeds only what's new, in one call.
        """
        masks = np.atleast_2d(np.asarray(masks, dtype=bool))
        self.n_requested += len(masks)

        keys = [m.tobytes() for m in masks]
        todo, todo_keys = [], []
        seen = set()
        for key, mask in zip(keys, masks):
            if key in self._cache or key in seen:
                continue
            seen.add(key)
            todo_keys.append(key)
            todo.append(render(self.question, self.units, mask))

        if todo:
            scores, candidates = self.retriever.score_query_variants(todo, doc_id=self.doc_id)
            if not self.candidates:
                self.candidates = candidates
            self.n_embedded += len(todo)
            for key, row in zip(todo_keys, scores):
                self._cache[key] = row

        return np.stack([self._cache[k] for k in keys])

    def base_scores(self) -> np.ndarray:
        """Scores of the unperturbed query -- the (1, n_candidates) reference row."""
        return self.score(self.full_mask())[0]

    def stats(self) -> dict:
        return {
            "n_units": self.n_units,
            "variants_requested": self.n_requested,
            "variants_embedded": self.n_embedded,
            "cache_hit_rate": round(1 - self.n_embedded / max(self.n_requested, 1), 4),
        }


# ---- value functions ----------------------------------------------------------------------

def score_value_fn(target_col: int):
    """v(S) = the similarity score of one target chunk under coalition S.

    Explains a single retrieved chunk. Attribution weights come out in score units (cosine
    similarity), so they are directly comparable to the base score and to each other.
    """
    def value(scores: np.ndarray) -> np.ndarray:
        return scores[:, target_col].astype(np.float64)
    return value


def ranking_value_fn(base_scores: np.ndarray, depth: int = 10, p: float = 0.9):
    """v(S) = rank-biased overlap between the ranking induced by coalition S and the
    unperturbed ranking. This is the RankingSHAP value function.

    Explains the ordering as a whole rather than one chunk, so it answers "which words made
    the retriever produce *this list*" -- the question that matters when the failure being
    diagnosed is a wrong ranking rather than one wrong row.
    """
    base_ranking = list(np.argsort(-base_scores)[:depth])

    def value(scores: np.ndarray) -> np.ndarray:
        out = np.empty(len(scores), dtype=np.float64)
        for i, row in enumerate(scores):
            out[i] = rbo(base_ranking, list(np.argsort(-row)[:depth]), p=p, depth=depth)
        return out
    return value
