"""Results dashboard -- every headline number, read from the same files the paper cites."""
from __future__ import annotations

import json

import streamlit as st

from common import RESULTS, _fmt


def render() -> None:
    st.markdown('<div class="ar-eyebrow">Results dashboard</div>', unsafe_allow_html=True)
    render_overview()


def render_overview() -> None:
    """Every headline number, read from the same files the paper cites."""
    st.subheader("Verifier master table — `eval/results/verifier_comparison.json`")
    st.caption("mean = what the verifier reports · separation = correct − wrong · "
               "specificity = targeted-removal drop − random-removal drop · "
               "provenance = grounded − ungrounded (deterministic operand reference). "
               "\\* p<0.05 \\*\\* p<0.01 \\*\\*\\* p<0.001, permutation tests.")
    comp = RESULTS / "verifier_comparison.json"
    if comp.exists():
        rows = []
        for dataset, entries in json.loads(comp.read_text()).items():
            for r in entries:
                rows.append({"dataset": dataset, "verifier": r["verifier"], "n": r["n"],
                             "mean": f"{r['mean']:.3f}",
                             "separation": _fmt(r["separation"], r["separation_p"]),
                             "specificity": _fmt(r["specificity"], r["specificity_p"])
                             + (f" (n={r['n_perturbed']})" if r["n_perturbed"] else ""),
                             "provenance": _fmt(r["provenance"], r["provenance_p"])})
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.warning("run `python eval/baselines/compare_verifiers.py`")

    var = RESULTS / "perturbation_variance.json"
    if var.exists():
        st.subheader("Perturbation audit across seeds — `perturbation_variance.json`")
        rows = []
        for dataset, r in json.loads(var.read_text()).items():
            rows.append({"dataset": dataset, "seeds": r["n_seeds"],
                         "specificity mean ± sd": f"{r['specificity_mean']:+.3f} ± {r['specificity_sd']:.3f}",
                         "range": f"{r['specificity_min']:+.3f} .. {r['specificity_max']:+.3f}",
                         "p<0.05 under every seed": r["all_seeds_significant_05"],
                         "deterministic arms identical": r["control_and_targeted_identical_across_seeds"]})
        st.dataframe(rows, width="stretch", hide_index=True)

    b3 = RESULTS / "b3_comparison.md"
    if b3.exists():
        st.subheader("B1 / B2 / B3 — `b3_comparison.md`")
        st.markdown(b3.read_text().split("\n", 1)[1])      # drop the H1, we have our own
    else:
        st.warning("run `python eval/baselines/build_b3_table.py`")

    fail = RESULTS / "failure_cases_summary.json"
    if fail.exists():
        st.subheader("Labeled verifier failure cases — `failure_cases_summary.json`")
        st.caption("FN = correct + operands grounded, scored unfaithful · FP = wrong with gold "
                   "retrieved (or operand missing), scored faithful · insensitive / nonspecific "
                   "from the perturbation audit · rows in `failure_cases_<dataset>_dev.jsonl`")
        rows = []
        for dataset, per in json.loads(fail.read_text())["labels"].items():
            for verifier, c in per.items():
                rows.append({"dataset": dataset, "verifier": verifier, "n": c["n"],
                             "FN": c["false_negative"], "FP": c["false_positive"],
                             "insensitive": c["insensitive"], "nonspecific": c["nonspecific"],
                             "unscorable": c["unscorable"], "any label": c["any_label"]})
        st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("Retrieval attribution (Contribution 1) — `attribution_*_dev_*.json`")
    st.caption("Lift over a random-unit baseline in normalized comprehensiveness at the top-20% "
               "of query units removed. There is no gold explanation, so only faithfulness "
               "against random is claimed.")
    rows = []
    for name in ("attribution_finqa_dev_score", "attribution_finqa_dev_ranking",
                 "attribution_tatqa_dev_score"):
        path = RESULTS / f"{name}.json"
        if path.exists():
            a = json.loads(path.read_text())
            for method, m in a["by_method"].items():
                rows.append({"file": name, "n": m["n"], "method": method,
                             "lift over random @0.2": f"{m['lift_over_random@0.2']:.3f}",
                             "stopword mass": f"{m['mean_mass_by_kind'].get('stopword', 0):.3f}",
                             "number mass": f"{m['mean_mass_by_kind'].get('number', 0):.3f}"})
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("B1 baseline — `b1_rag_*_dev.json`, `b1_retrieval_*_dev.json`")
    rows = []
    for dataset in ("finqa", "tatqa"):
        b1 = RESULTS / f"b1_rag_{dataset}_dev.json"
        rt = RESULTS / f"b1_retrieval_{dataset}_dev.json"
        if b1.exists():
            r = json.loads(b1.read_text())
            ret = json.loads(rt.read_text())["overall"] if rt.exists() else r["overall"]
            a = r["answers"]
            rows.append({"dataset": dataset, "recall@5": f"{ret['recall@5']:.3f}",
                         "MRR": f"{ret['mrr']:.3f}",
                         "numeric accuracy": f"{a['numeric_accuracy']:.3f}",
                         "span F1": f"{a['span_token_f1']:.3f}" if a["n_textual"] >= 10 else "—",
                         "citation precision": f"{r['citation_precision']:.3f}",
                         "program rate": f"{r['program_rate']:.3f}",
                         "model": r["config"]["generation_model"].split("/")[-1]})
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)


# ---- page ---------------------------------------------------------------------------------

