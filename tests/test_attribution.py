"""
Tests for Month 4 / Contribution 1 (src/attribution/) and eval/metrics/attribution_metrics.py.

Run with:  pytest tests/test_attribution.py -v

Offline and fast: the retriever is built on `HashEmbedder` over the checked-in FinQA fixture,
so nothing downloads and the whole file runs in about a second.

The most valuable test here is `test_shapley_satisfies_the_efficiency_axiom`. Shapley values
are *defined* by their axioms, so the axiom is a genuine correctness oracle -- if the
contributions don't sum to v(full) - v(empty), the implementation is wrong regardless of how
plausible the output looks. That's a much stronger check than eyeballing whether the top-
weighted word seems sensible, which is the trap with explanation code: it produces
confident-looking numbers whether or not the maths is right.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "src" / "ingestion", ROOT / "src" / "retrieval",
           ROOT / "src" / "attribution", ROOT / "eval" / "metrics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from attribution_metrics import faithfulness, mass_by_kind, spearman  # noqa: E402
from attributors import (  # noqa: E402
    OcclusionAttributor,
    ShapleyAttributor,
    SurrogateAttributor,
    build_attributor,
)
from chunk_index import ChunkIndex  # noqa: E402
from embedder import HashEmbedder  # noqa: E402
from explainer import explain_query  # noqa: E402
from finqa_loader import build_chunk_index, load_finqa  # noqa: E402
from perturbation import (  # noqa: E402
    PerturbationScorer,
    ranking_value_fn,
    rbo,
    score_value_fn,
)
from retriever import Retriever  # noqa: E402
from segmentation import render, segment_query  # noqa: E402

FINQA_SAMPLE = ROOT / "data" / "sample" / "finqa_sample.json"


@pytest.fixture(scope="module")
def sample():
    chunks, questions = [], []
    for c, q in load_finqa(str(FINQA_SAMPLE)):
        chunks.extend(c)
        questions.append(q)
    return build_chunk_index(chunks), questions


@pytest.fixture(scope="module")
def retriever(sample):
    chunks, _ = sample
    index = ChunkIndex.build(chunks, HashEmbedder())
    return Retriever(embedder=HashEmbedder(), index=index, top_k=5, scope="document")


@pytest.fixture
def scored(retriever, sample):
    """A scorer plus a score-based value function for the top chunk of one real question."""
    _, questions = sample
    question = questions[0]
    units = segment_query(question.question)
    scorer = PerturbationScorer(retriever, question.question, units, question.doc_id)
    base = scorer.base_scores()
    return scorer, units, score_value_fn(int(np.argmax(base)))


# ---- segmentation ------------------------------------------------------------------------

def test_segmentation_labels_numbers_entities_and_stopwords():
    units = segment_query("What was the change in Transportation Solutions revenue in 2019?")
    kinds = {u.text: u.kind for u in units}
    assert kinds["2019"] == "number"
    assert kinds["Transportation Solutions"] == "entity", "multi-word entity must stay one unit"
    assert kinds["the"] == "stopword"
    assert kinds["revenue"] == "term"


def test_currency_amounts_are_one_numeric_unit():
    units = segment_query("what was the total of $1,496.5 million?")
    assert any(u.text == "$1,496.5" and u.kind == "number" for u in units)


def test_units_are_ordered_and_non_overlapping():
    units = segment_query("what is the percentage change in net revenue from 2018 to 2019?")
    assert [u.index for u in units] == list(range(len(units)))
    for a, b in zip(units, units[1:]):
        assert a.end <= b.start, "unit spans must not overlap"


def test_render_blanks_spans_and_preserves_the_rest():
    """Perturbation must change exactly one thing. Re-joining surviving tokens with spaces
    would also strip punctuation, and the resulting score drop would be partly an artifact
    of that second change rather than of the removed unit."""
    q = "what was the change in net revenue?"
    units = segment_query(q)
    mask = [u.text != "revenue" for u in units]
    rendered = render(q, units, mask)
    assert "revenue" not in rendered
    assert rendered == "what was the change in net ?"        # punctuation kept, spaces collapsed
    assert render(q, units, [True] * len(units)) == q         # full mask is a no-op
    assert render(q, units, [False] * len(units)) == "?"


def test_segmentation_survives_without_the_spacy_model():
    """The spaCy model is an optional download. Missing it must degrade entity *grouping*,
    never break attribution."""
    assert segment_query("what was the Transportation Solutions total?", use_ner=False)


# ---- perturbation engine ------------------------------------------------------------------

def test_scorer_batches_and_caches(scored):
    # the fixture already scored the full mask, so measure deltas rather than absolutes
    scorer, units, _ = scored
    embedded_before, requested_before = scorer.n_embedded, scorer.n_requested

    masks = np.array([scorer.full_mask(), scorer.empty_mask(), scorer.full_mask()])
    scores = scorer.score(masks)

    assert scores.shape == (3, len(scorer.candidates))
    assert np.allclose(scores[0], scores[2]), "identical masks must give identical scores"
    assert scorer.n_requested - requested_before == 3
    assert scorer.n_embedded - embedded_before == 1, (
        "only the empty mask is new -- the repeated full mask must come from cache"
    )


def test_cache_persists_across_calls(scored):
    scorer, _, _ = scored
    scorer.score(scorer.full_mask())
    embedded_after_first = scorer.n_embedded
    scorer.score(scorer.full_mask())
    assert scorer.n_embedded == embedded_after_first


def test_rbo_is_normalized_and_top_weighted():
    assert rbo([1, 2, 3], [1, 2, 3], depth=3) == pytest.approx(1.0)
    assert rbo([1, 2, 3], [4, 5, 6], depth=3) == pytest.approx(0.0)
    # a swap at the top hurts more than the same swap further down
    top_swap = rbo([1, 2, 3, 4], [2, 1, 3, 4], depth=4)
    deep_swap = rbo([1, 2, 3, 4], [1, 2, 4, 3], depth=4)
    assert top_swap < deep_swap


def test_ranking_value_fn_peaks_on_the_unperturbed_ranking(scored):
    scorer, _, _ = scored
    base = scorer.base_scores()
    value_fn = ranking_value_fn(base, depth=5)
    values = value_fn(scorer.score(np.array([scorer.full_mask(), scorer.empty_mask()])))
    assert values[0] == pytest.approx(1.0), "the full query reproduces its own ranking"
    assert values[1] <= values[0]


# ---- the axiom test -----------------------------------------------------------------------

def test_shapley_satisfies_the_efficiency_axiom(scored):
    """Sum of contributions must equal v(full) - v(empty), exactly. This is the definition of
    a Shapley value, so it is a real oracle rather than a plausibility check."""
    scorer, units, value_fn = scored
    v_full = value_fn(scorer.score(scorer.full_mask()))[0]
    v_empty = value_fn(scorer.score(scorer.empty_mask()))[0]

    exact = ShapleyAttributor(exact_max_units=99).attribute(scorer, value_fn)
    assert exact.meta["mode"] == "exact"
    assert exact.weights.sum() == pytest.approx(v_full - v_empty, abs=1e-9)


def test_monte_carlo_shapley_matches_exact(scored):
    """The sampled estimator is what the real runs use, so it has to be checked against the
    exact one -- not just against itself."""
    scorer, _, value_fn = scored
    exact = ShapleyAttributor(exact_max_units=99).attribute(scorer, value_fn)
    sampled = ShapleyAttributor(n_permutations=300, exact_max_units=0).attribute(
        scorer, value_fn)

    assert sampled.meta["mode"] == "sampled"
    # permutation sampling preserves efficiency exactly, whatever the sample size
    assert sampled.weights.sum() == pytest.approx(exact.weights.sum(), abs=1e-9)
    # Compared by value, not by rank. Under the bag-of-words test embedder several units are
    # *exactly* tied (every stopword absent from the chunk contributes identically), and rank
    # correlation then penalizes the sampled estimator for breaking those ties in some
    # arbitrary order -- an ordering that carries no information either way.
    assert np.abs(exact.weights - sampled.weights).max() < 0.02


def test_shapley_gives_zero_to_a_unit_that_changes_nothing(scored):
    """The null-player axiom. A unit the value function ignores must get exactly 0."""
    scorer, units, _ = scored
    constant = lambda scores: np.ones(len(scores))  # noqa: E731 -- every coalition is equal
    result = ShapleyAttributor(exact_max_units=99).attribute(scorer, constant)
    assert np.allclose(result.weights, 0.0, atol=1e-9)


# ---- the three methods --------------------------------------------------------------------

@pytest.mark.parametrize("method", ["occlusion", "shapley", "surrogate"])
def test_every_method_returns_one_weight_per_unit(scored, method):
    scorer, units, value_fn = scored
    result = build_attributor(method).attribute(scorer, value_fn)
    assert len(result.weights) == len(units)
    assert np.isfinite(result.weights).all()


def test_methods_broadly_agree(scored):
    """They shouldn't be identical -- they measure different things -- but if they disagree
    entirely then "the explanation" isn't well defined and the eval would be reporting noise."""
    scorer, _, value_fn = scored
    occlusion = OcclusionAttributor().attribute(scorer, value_fn)
    shapley = ShapleyAttributor(exact_max_units=99).attribute(scorer, value_fn)
    surrogate = SurrogateAttributor(n_samples=300).attribute(scorer, value_fn)
    assert spearman(occlusion.weights, shapley.weights) > 0.5
    assert spearman(shapley.weights, surrogate.weights) > 0.5


def test_attributors_are_deterministic(scored):
    """Same seed, same weights. A stochastic explanation can't be compared across B1/B2/B3."""
    scorer, _, value_fn = scored
    a = ShapleyAttributor(n_permutations=40, seed=7, exact_max_units=0).attribute(
        scorer, value_fn)
    b = ShapleyAttributor(n_permutations=40, seed=7, exact_max_units=0).attribute(
        scorer, value_fn)
    assert np.allclose(a.weights, b.weights)


def test_unknown_method_fails_fast():
    with pytest.raises(ValueError, match="unknown attribution method"):
        build_attributor("shap")


# ---- explainer ----------------------------------------------------------------------------

def test_explain_query_end_to_end(retriever, sample):
    _, questions = sample
    question = questions[0]
    explanation = explain_query(retriever, question.question, doc_id=question.doc_id,
                                method="occlusion", top_k=2)
    assert len(explanation.chunks) == 2
    assert [c.rank for c in explanation.chunks] == [0, 1]
    assert explanation.chunks[0].base_score >= explanation.chunks[1].base_score
    assert len(explanation.chunks[0].attributions) == len(explanation.units)
    assert "cache_hit_rate" in explanation.stats
    assert question.question in explanation.summary()


def test_explaining_more_chunks_costs_no_extra_embeddings(retriever, sample):
    """The coalitions depend only on the query, and one matmul scores every chunk -- so
    targets 2..k are column reads out of the cache. This is what makes explaining the whole
    retrieved top-k the default instead of a luxury."""
    _, questions = sample
    question = questions[0]
    one = explain_query(retriever, question.question, doc_id=question.doc_id,
                        method="occlusion", top_k=1)
    four = explain_query(retriever, question.question, doc_id=question.doc_id,
                         method="occlusion", top_k=4)
    assert len(four.chunks) == 4
    assert four.stats["variants_embedded"] == one.stats["variants_embedded"]


def test_ranking_mode_explains_the_ordering_not_a_chunk(retriever, sample):
    _, questions = sample
    question = questions[0]
    explanation = explain_query(retriever, question.question, doc_id=question.doc_id,
                                method="occlusion", mode="ranking")
    assert explanation.ranking_attributions and not explanation.chunks
    assert len(explanation.ranking_attributions) == len(explanation.units)


def test_bad_mode_fails_fast(retriever, sample):
    _, questions = sample
    with pytest.raises(ValueError, match="mode must be"):
        explain_query(retriever, questions[0].question, doc_id=questions[0].doc_id,
                      mode="sideways")


def test_empty_query_does_not_crash(retriever, sample):
    """A query of pure punctuation has no units. It must return an empty explanation, not
    divide by zero somewhere deep in the estimator."""
    _, questions = sample
    explanation = explain_query(retriever, "???", doc_id=questions[0].doc_id,
                                method="shapley")
    assert explanation.units == [] and explanation.chunks == []


# ---- attribution metrics ------------------------------------------------------------------

def test_faithfulness_reports_a_random_baseline(scored):
    """Removing any 20% of a query lowers the score somewhat, so comprehensiveness alone
    proves nothing -- the random comparison is what makes the number meaningful."""
    scorer, _, value_fn = scored
    weights = OcclusionAttributor().attribute(scorer, value_fn).weights
    result = faithfulness(scorer, value_fn, weights, fractions=(0.2,))
    for key in ("comprehensiveness@0.2", "random_comprehensiveness@0.2",
                "sufficiency@0.2", "lift_over_random@0.2", "span"):
        assert key in result
    assert np.isfinite(result["comprehensiveness@0.2"])


def test_faithfulness_prefers_real_weights_over_reversed_ones(scored):
    """Sanity with teeth: removing the units a method calls *most* important must hurt more
    than removing the ones it calls least important. If this fails, the sign convention is
    inverted somewhere."""
    scorer, _, value_fn = scored
    weights = ShapleyAttributor(exact_max_units=99).attribute(scorer, value_fn).weights
    forward = faithfulness(scorer, value_fn, weights, fractions=(0.3,))
    reversed_ = faithfulness(scorer, value_fn, -weights, fractions=(0.3,))
    assert forward["comprehensiveness@0.3"] > reversed_["comprehensiveness@0.3"]


def test_mass_by_kind_is_a_distribution():
    units = segment_query("what was the change in net revenue in 2019?")
    weights = np.ones(len(units))
    shares = mass_by_kind(units, weights)
    assert shares and abs(sum(shares.values()) - 1.0) < 1e-6
    assert set(shares) <= {"number", "entity", "term", "stopword"}


def test_mass_by_kind_ignores_negative_weights():
    """A negative weight means the unit pushed the chunk *down* -- a different phenomenon.
    Summing it in would cancel real signal and understate the mass on other kinds."""
    units = segment_query("net revenue 2019")
    weights = np.array([1.0, -5.0, 1.0])[:len(units)]
    shares = mass_by_kind(units, weights)
    assert abs(sum(shares.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in shares.values())


def test_spearman_handles_degenerate_input():
    assert spearman(np.array([1.0]), np.array([2.0])) is None       # too short
    assert spearman(np.zeros(5), np.arange(5.0)) is None            # no variance
    assert spearman(np.arange(5.0), np.arange(5.0)) == pytest.approx(1.0)
    assert spearman(np.arange(5.0), -np.arange(5.0)) == pytest.approx(-1.0)
