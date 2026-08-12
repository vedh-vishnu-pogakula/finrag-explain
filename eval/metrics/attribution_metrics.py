"""
How do you evaluate an explanation when no dataset says what the right explanation is?

This is the central methodological problem of Contribution 1, and CLAUDE.md already accepts
it as a fact of the field rather than a gap to paper over: no financial QA dataset labels
which *query tokens* should have driven retrieval. So attribution is not scored against a gold
explanation. It is scored on **faithfulness** -- whether the explanation's own claims hold up
when acted on:

  Comprehensiveness -- if these units really drove the score, removing them should collapse
                       it. Measured as v(full) - v(full minus the top-weighted units).
                       **Higher is better.**

  Sufficiency       -- if these units really drove the score, they alone should nearly
                       reproduce it. Measured as v(full) - v(only the top-weighted units).
                       **Lower is better** (0 = the top units recover the score exactly).
                       It can legitimately go *negative*: dropping the rest of the query
                       sometimes scores higher than the full query, because the discarded
                       words were diluting the embedding. That is a real property of dense
                       retrieval, not a bug in the metric, and it is worth reporting as such.

Both are normalized by the query's total attributable range, v(full) - v(empty), so a question
whose score barely moves under any perturbation doesn't dominate the average purely because
its scores are large.

**The random baseline is not optional.** Removing any 20% of a query lowers its score
somewhat, so a comprehensiveness number in isolation proves nothing. Every metric here is
reported alongside the same measurement with randomly chosen units, and the honest claim is
the *gap* between them. A method that fails to beat random is not explaining anything --
and reporting that outcome, if it happens, is a result rather than a failure.

This is the same logic Contribution 2 later applies to RAGAS: don't trust a score, perturb
what it claims matters and check that it moves the way it should. Contribution 1 is where that
argument gets built and validated on something we control.
"""
from __future__ import annotations

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared -- the ranking Spearman needs."""
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # average tied groups so tied units don't get an arbitrary ordering
    sorted_values = values[order]
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Rank correlation between two attribution vectors. Used to report how much the three
    methods actually agree -- if they disagree wildly, "the explanation" isn't well defined
    and that has to be said out loud."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(a) != len(b):
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _top_k_mask(weights: np.ndarray, k: int, keep_top: bool) -> np.ndarray:
    """Mask keeping (or removing) the k highest-weighted units."""
    top = np.argsort(-weights)[:k]
    mask = np.zeros(len(weights), dtype=bool) if keep_top else np.ones(len(weights), dtype=bool)
    mask[top] = keep_top
    return mask


def faithfulness(scorer, value_fn, weights: np.ndarray,
                 fractions: tuple = (0.2, 0.5), seed: int = 13) -> dict:
    """Comprehensiveness and sufficiency at each fraction, with matched random baselines.

    All the masks needed are built first and scored in one batched call, and they mostly hit
    the cache the attribution run already populated -- so evaluating an explanation costs
    close to nothing on top of producing it.
    """
    weights = np.asarray(weights, dtype=np.float64)
    n_units = len(weights)
    if n_units == 0:
        return {}

    rng = np.random.default_rng(seed)
    full = np.ones(n_units, dtype=bool)
    empty = np.zeros(n_units, dtype=bool)

    masks, labels = [full, empty], ["full", "empty"]
    for fraction in fractions:
        k = max(1, int(round(fraction * n_units)))
        random_top = rng.permutation(n_units)[:k]
        random_weights = np.zeros(n_units)
        random_weights[random_top] = 1.0

        masks += [
            _top_k_mask(weights, k, keep_top=False),          # comprehensiveness
            _top_k_mask(weights, k, keep_top=True),           # sufficiency
            _top_k_mask(random_weights, k, keep_top=False),   # random comprehensiveness
            _top_k_mask(random_weights, k, keep_top=True),    # random sufficiency
        ]
        labels += [f"comp@{fraction}", f"suff@{fraction}",
                   f"rand_comp@{fraction}", f"rand_suff@{fraction}"]

    values = value_fn(scorer.score(np.array(masks)))
    by_label = dict(zip(labels, values))

    v_full, v_empty = by_label["full"], by_label["empty"]
    span = v_full - v_empty
    # A query whose score is unmoved by removing everything has no attributable signal at all;
    # normalizing by ~0 would produce meaningless spikes, so those questions are reported as
    # raw-only and excluded from normalized aggregates.
    normalizable = abs(span) > 1e-9

    out = {"v_full": float(v_full), "v_empty": float(v_empty), "span": float(span),
           "normalizable": bool(normalizable)}
    for fraction in fractions:
        comp = v_full - by_label[f"comp@{fraction}"]
        suff = v_full - by_label[f"suff@{fraction}"]
        rand_comp = v_full - by_label[f"rand_comp@{fraction}"]
        rand_suff = v_full - by_label[f"rand_suff@{fraction}"]
        out[f"comprehensiveness@{fraction}"] = float(comp)
        out[f"sufficiency@{fraction}"] = float(suff)
        out[f"random_comprehensiveness@{fraction}"] = float(rand_comp)
        out[f"random_sufficiency@{fraction}"] = float(rand_suff)
        if normalizable:
            out[f"norm_comprehensiveness@{fraction}"] = float(comp / span)
            out[f"norm_sufficiency@{fraction}"] = float(suff / span)
            out[f"norm_random_comprehensiveness@{fraction}"] = float(rand_comp / span)
            # the headline: how far the method's comprehensiveness beats picking units blindly
            out[f"lift_over_random@{fraction}"] = float((comp - rand_comp) / span)
    return out


def mass_by_kind(units: list, weights: np.ndarray) -> dict:
    """Share of total *positive* attribution mass carried by each unit kind.

    This is what turns attribution into a claim about financial QA rather than a per-query
    curiosity: "numeric values carry N% of retrieval attribution mass on FinQA" is a finding.
    Negative weights are excluded -- they mean the unit was actively pulling the chunk down,
    which is a different phenomenon and would cancel out real signal if summed in.
    """
    weights = np.asarray(weights, dtype=np.float64)
    positive = np.clip(weights, 0, None)
    total = positive.sum()
    if total <= 0:
        return {}
    shares: dict = {}
    for unit, weight in zip(units, positive):
        shares[unit.kind] = shares.get(unit.kind, 0.0) + float(weight)
    return {kind: round(value / total, 4) for kind, value in sorted(shares.items())}


def aggregate_attribution(records: list) -> dict:
    """Roll per-question attribution records into the report written to eval/results/."""
    if not records:
        return {}

    def mean_of(key, rows):
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(values)), 4) if values else None

    by_method: dict = {}
    for record in records:
        for method, payload in record["methods"].items():
            by_method.setdefault(method, []).append(payload)

    report: dict = {"n_questions": len(records), "by_method": {}}
    for method, rows in by_method.items():
        faith_rows = [r["faithfulness"] for r in rows if r.get("faithfulness")]
        entry: dict = {"n": len(rows)}
        for key in sorted({k for r in faith_rows for k in r}):
            if key.startswith(("norm_", "lift_", "comprehensiveness", "sufficiency",
                               "random_")):
                entry[key] = mean_of(key, faith_rows)
        # Average over *all* questions, counting a kind the query didn't contain as 0.0.
        # Averaging only over questions where a kind appears looks reasonable and is wrong:
        # the resulting numbers sum to well above 1 and can't be read as shares of mass.
        # `presence_rate` keeps the information that would otherwise be lost -- how often a
        # kind occurs at all -- without corrupting the distribution.
        all_kinds = sorted({k for r in rows for k in (r.get("mass_by_kind") or {})})
        scored_rows = [r for r in rows if r.get("mass_by_kind")]
        entry["mean_mass_by_kind"] = {
            kind: round(float(np.mean([(r["mass_by_kind"]).get(kind, 0.0)
                                       for r in scored_rows])), 4)
            for kind in all_kinds
        } if scored_rows else {}
        entry["kind_presence_rate"] = {
            kind: round(float(np.mean([kind in (r.get("mass_by_kind") or {})
                                       for r in rows])), 4)
            for kind in all_kinds
        }
        entry["mean_units_per_query"] = mean_of("n_units", rows)
        report["by_method"][method] = entry

    agreements: dict = {}
    for record in records:
        for pair, value in (record.get("agreement") or {}).items():
            if value is not None:
                agreements.setdefault(pair, []).append(value)
    report["method_agreement_spearman"] = {
        pair: round(float(np.mean(values)), 4) for pair, values in sorted(agreements.items())
    }
    return report
