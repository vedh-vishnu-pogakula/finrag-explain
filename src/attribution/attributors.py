"""
The three attribution estimators of Contribution 1.

CLAUDE.md is explicit that this must be anchored on RankingSHAP / Rank-LIME rather than being
"a naive off-the-shelf SHAP/LIME call", so all three are implemented directly against the
retriever's own similarity function. That is not purism -- an off-the-shelf explainer expects
a classifier `predict_proba` over a fixed feature space, and neither the feature space (the
units of *this* query) nor the output (a similarity, or a whole ranking) fits that shape.
Implementing them here is also what makes the batched single-matmul design possible, which is
what keeps the month computationally feasible.

    Occlusion   -- leave-one-out. M+1 variants. The cheap baseline every attribution paper
                   reports, and the one to beat.
    Shapley     -- RankingSHAP-anchored. The only method here with axiomatic guarantees:
                   contributions sum exactly to v(full) - v(empty), and units that never
                   change the value get exactly zero.
    Surrogate   -- Rank-LIME-anchored. Fits a locally-weighted linear model over sampled
                   coalitions; the coefficients are the attributions.

All three take the same `(scorer, value_fn)` pair, so any of them can explain either a single
chunk's score or the whole ranking (see `perturbation.py`) without changing a line here. That
separation is what lets the evaluation compare methods and value functions independently.

**Why occlusion is not enough on its own** -- and this is the answer to "why not just do
leave-one-out": occlusion measures each unit's effect only in the presence of *all* the
others, so it systematically under-credits redundant units. In "change in net revenue from
2018 to 2019", removing "revenue" alone barely moves the score because "net" still retrieves
the same rows; removing both collapses it. Occlusion reports both as unimportant. Shapley
averages over all orderings and splits the shared credit between them.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import factorial

import numpy as np

from perturbation import PerturbationScorer


@dataclass
class AttributionResult:
    weights: np.ndarray
    method: str
    meta: dict = field(default_factory=dict)


class OcclusionAttributor:
    """Leave-one-out: weight_i = v(full) - v(full without unit i).

    Positive weight = removing the unit hurt, i.e. the unit was pulling the score up.
    """

    name = "occlusion"

    def attribute(self, scorer: PerturbationScorer, value_fn) -> AttributionResult:
        m = scorer.n_units
        masks = np.ones((m + 1, m), dtype=bool)
        for i in range(m):
            masks[i + 1, i] = False           # row 0 is the unperturbed query
        values = value_fn(scorer.score(masks))
        return AttributionResult(
            weights=values[0] - values[1:],
            method=self.name,
            meta={"n_coalitions": m + 1},
        )


class ShapleyAttributor:
    """RankingSHAP-style Shapley values over query units.

    Exact enumeration is used for short queries (2^M coalitions), Monte Carlo permutation
    sampling otherwise. The switch is automatic because exactness is free when it's cheap and
    intractable when it isn't -- and having both means the sampled estimator can be tested
    against the exact one rather than only against itself.

    Every coalition needed across every sampled permutation is collected before anything is
    scored, so the whole estimate is one batched embedding pass.
    """

    name = "shapley"

    def __init__(self, n_permutations: int = 64, seed: int = 13, exact_max_units: int = 10):
        self.n_permutations = n_permutations
        self.seed = seed
        self.exact_max_units = exact_max_units

    def attribute(self, scorer: PerturbationScorer, value_fn) -> AttributionResult:
        m = scorer.n_units
        if m == 0:
            return AttributionResult(np.zeros(0), self.name, {"mode": "empty"})
        if m <= self.exact_max_units:
            return self._exact(scorer, value_fn, m)
        return self._sampled(scorer, value_fn, m)

    def _exact(self, scorer, value_fn, m) -> AttributionResult:
        masks = np.array(list(itertools.product([False, True], repeat=m)), dtype=bool)
        values = value_fn(scorer.score(masks))
        lookup = {mask.tobytes(): v for mask, v in zip(masks, values)}

        phi = np.zeros(m, dtype=np.float64)
        for i in range(m):
            for mask, v_without in zip(masks, values):
                if mask[i]:
                    continue
                s = int(mask.sum())
                weight = factorial(s) * factorial(m - s - 1) / factorial(m)
                with_i = mask.copy()
                with_i[i] = True
                phi[i] += weight * (lookup[with_i.tobytes()] - v_without)
        return AttributionResult(phi, self.name,
                                 {"mode": "exact", "n_coalitions": int(2 ** m)})

    def _sampled(self, scorer, value_fn, m) -> AttributionResult:
        rng = np.random.default_rng(self.seed)
        perms = [rng.permutation(m) for _ in range(self.n_permutations)]

        # Build every prefix coalition for every permutation up front: one batch, not P.
        masks = np.empty((len(perms) * (m + 1), m), dtype=bool)
        row = 0
        for perm in perms:
            live = np.zeros(m, dtype=bool)
            masks[row] = live
            row += 1
            for unit in perm:
                live[unit] = True
                masks[row] = live
                row += 1

        values = value_fn(scorer.score(masks)).reshape(len(perms), m + 1)

        phi = np.zeros(m, dtype=np.float64)
        for perm, vals in zip(perms, values):
            phi[perm] += np.diff(vals)        # marginal contribution in this ordering
        phi /= len(perms)
        return AttributionResult(phi, self.name, {
            "mode": "sampled",
            "n_permutations": self.n_permutations,
            "n_coalitions": int(masks.shape[0]),
        })


class SurrogateAttributor:
    """Rank-LIME-style local surrogate: fit a weighted ridge regression over sampled
    coalitions and read the coefficients as attributions.

    Samples are drawn across the whole sparsity range (keep 1 unit, keep 2, ... keep all)
    rather than the usual independent coin-flip per feature. On a 10-word query a coin flip
    concentrates every sample near half the words removed, so the surrogate never observes
    what a near-complete query does -- which is exactly the neighbourhood it is supposed to be
    local to.

    The kernel weights samples by how much of the query survives, so coalitions close to the
    original dominate the fit. The intercept is left unregularized: shrinking it would push
    the model's baseline toward zero and bias every coefficient upward to compensate.
    """

    name = "surrogate"

    def __init__(self, n_samples: int = 256, seed: int = 13, kernel_width: float = 0.35,
                 ridge_alpha: float = 1.0):
        self.n_samples = n_samples
        self.seed = seed
        self.kernel_width = kernel_width
        self.ridge_alpha = ridge_alpha

    def attribute(self, scorer: PerturbationScorer, value_fn) -> AttributionResult:
        m = scorer.n_units
        if m == 0:
            return AttributionResult(np.zeros(0), self.name, {"n_samples": 0})

        rng = np.random.default_rng(self.seed)
        masks = np.zeros((self.n_samples, m), dtype=bool)
        masks[0] = True                                   # anchor the fit on the full query
        for i in range(1, self.n_samples):
            n_keep = rng.integers(1, m + 1)
            masks[i, rng.choice(m, size=n_keep, replace=False)] = True

        values = value_fn(scorer.score(masks))

        frac_kept = masks.sum(axis=1) / m
        sample_weights = np.exp(-((1.0 - frac_kept) ** 2) / (self.kernel_width ** 2))

        design = np.hstack([np.ones((len(masks), 1)), masks.astype(np.float64)])
        penalty = np.eye(design.shape[1]) * self.ridge_alpha
        penalty[0, 0] = 0.0                               # never regularize the intercept
        gram = design.T @ (sample_weights[:, None] * design) + penalty
        target = design.T @ (sample_weights * values)
        coefficients = np.linalg.solve(gram, target)

        return AttributionResult(coefficients[1:], self.name, {
            "n_samples": self.n_samples,
            "n_coalitions": int(len(masks)),
            "kernel_width": self.kernel_width,
            "ridge_alpha": self.ridge_alpha,
        })


ATTRIBUTORS = {
    "occlusion": OcclusionAttributor,
    "shapley": ShapleyAttributor,
    "surrogate": SurrogateAttributor,
}


def build_attributor(method: str, **kwargs):
    if method not in ATTRIBUTORS:
        raise ValueError(f"unknown attribution method {method!r} "
                         f"(choose from {sorted(ATTRIBUTORS)})")
    return ATTRIBUTORS[method](**kwargs)
