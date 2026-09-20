"""Failure-case browser -- the 1,500 labeled (question, verifier) verdicts, filterable."""
from __future__ import annotations

import json

import streamlit as st

from common import RESULTS, pill, verifier_style

LABELS = {
    "false_negative": "false negative — correct answer, every operand in the evidence, scored unfaithful",
    "false_positive": "false positive — wrong (gold evidence retrieved) or operand missing, scored faithful",
    "insensitive": "insensitive — removing the operand sentence did not lower a non-zero score",
    "nonspecific": "nonspecific — targeted drop ≤ random drop",
    "unscorable": "unscorable — the judge produced no statements",
}


@st.cache_data(show_spinner=False)
def load_rows(dataset: str) -> list:
    p = RESULTS / f"failure_cases_{dataset}_dev.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def render() -> None:
    st.markdown('<div class="ar-eyebrow">Failure cases — Contribution 2\'s dataset</div>',
                unsafe_allow_html=True)
    st.markdown(pill("replay", "labels are mechanical: derived from result files, reproducible to the row"),
                unsafe_allow_html=True)
    st.caption("`python eval/baselines/export_failure_cases.py` → `eval/results/failure_cases_"
               "{finqa,tatqa}_dev.jsonl`. One row per (question, verifier); a row with no label is a "
               "case where the verifier behaved as a faithfulness metric should.")

    dataset = st.sidebar.radio("Dataset", ["finqa", "tatqa"], horizontal=True, key="dataset")
    rows = load_rows(dataset)
    if not rows:
        st.warning("no failure-case file for this dataset")
        return
    verifiers = sorted({r["verifier"] for r in rows},
                       key=lambda v: ["NLI", "Phi", "Qwen"].index(next(
                           (k for k in ("NLI", "Phi", "Qwen") if k in verifier_style(v)[0]), "Qwen")))
    vsel = st.sidebar.multiselect("Verifier", verifiers, default=verifiers,
                                  format_func=lambda v: verifier_style(v)[0])
    lsel = st.sidebar.multiselect("Label", list(LABELS), default=["false_negative", "false_positive"],
                                  format_func=lambda k: k.replace("_", " "))
    only_restated = st.sidebar.selectbox("Answer value", ["any", "restates a figure in the evidence",
                                                          "computed (value not in evidence)"])
    sub = [r for r in rows if r["verifier"] in vsel and (not lsel or any(l in r["labels"] for l in lsel))]
    if only_restated.startswith("restates"):
        sub = [r for r in sub if r["restates_figure"] is True]
    elif only_restated.startswith("computed"):
        sub = [r for r in sub if r["restates_figure"] is False]

    # counts for the current filter
    st.markdown(f"**{len(sub)}** rows match · " + " · ".join(
        f"{verifier_style(v)[0]}: {sum(1 for r in sub if r['verifier'] == v)}" for v in vsel))
    with st.expander("What the labels mean"):
        for k, v in LABELS.items():
            st.markdown(f"- **{k.replace('_', ' ')}** — {v.split(' — ', 1)[1]}")

    table = [{"question": r["question"][:80], "verifier": verifier_style(r["verifier"])[0],
              "gold": str(r["gold_answer"])[:14], "predicted": str(r["predicted"])[:14],
              "faithfulness": r["faithfulness"], "correct": "✅" if r["correct"] else "❌",
              "operands grounded": {True: "✅", False: "❌", None: "—"}[r["operands_grounded"]],
              "restates": {True: "yes", False: "no", None: "—"}[r["restates_figure"]],
              "labels": ", ".join(r["labels"]), "qa_id": r["qa_id"]} for r in sub]
    event = st.dataframe(table, width="stretch", hide_index=True, height=360,
                         on_select="rerun", selection_mode="single-row",
                         column_config={"qa_id": None})
    picked = event.selection.rows[0] if event and event.selection.rows else 0
    if not sub:
        return
    r = sub[picked]
    st.markdown(f"#### {r['question']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("gold", str(r["gold_answer"]))
    c2.metric("predicted", str(r["predicted"]))
    c3.metric(f"faithfulness · {verifier_style(r['verifier'])[0]}",
              "—" if r["faithfulness"] is None else f"{r['faithfulness']:.2f}")
    if r["answer_expression"]:
        st.code(f"answer_expression = {r['answer_expression']}", language="text")
    if r["labels"]:
        st.error("labels: " + ", ".join(r["labels"]))
    else:
        st.success("no label — the verifier behaved as it should on this question")
    left, right = st.columns(2)
    with left:
        st.markdown("**Statements and verdicts**")
        for s_, v in zip(r["statements"], r["verdicts"]):
            st.markdown(f"{'✅' if v else '❌'} {s_}")
        st.markdown(f"correct: {'yes' if r['correct'] else 'no'} · failure mode: "
                    f"`{r['failure_mode']}` · operands grounded: {r['operands_grounded']} · "
                    f"answer restates a figure in the evidence: {r['restates_figure']}")
    with right:
        if r["perturbation"]:
            p = r["perturbation"]
            st.markdown("**Perturbation scores (this verifier)**")
            st.markdown(f"control {p.get('control')} · targeted {p.get('targeted')} · random {p.get('random')}")
            if r["removed_targeted"]:
                st.markdown("removed (targeted):")
                for t in r["removed_targeted"]:
                    st.markdown(f"- {t[:160]}")
        else:
            st.caption("no perturbation run for this (question, verifier)")
    st.markdown(f'<div class="ar-source">row: eval/results/failure_cases_{dataset}_dev.jsonl · qa_id {r["qa_id"]}</div>',
                unsafe_allow_html=True)
