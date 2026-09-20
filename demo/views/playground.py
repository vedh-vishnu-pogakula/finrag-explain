"""
Audit playground -- remove evidence sentences yourself and watch the verifier react.

This is Contribution 2 made interactive. The NLI verifier is a local cross-encoder, so every
change re-verifies the cached statements against the surviving context in about a second.
The LLM verifier cannot run on the laptop; for questions in the 80-question LLM audit subset
its recorded scores for the same three conditions are shown beside the live one.
"""
from __future__ import annotations

import random
import sys

import plotly.graph_objects as go
import streamlit as st

from common import (CKPT, PLOTLY_TEMPLATE, ROOT, load_artifacts, pick_question, pill,
                    sidebar_dataset, verifier_style)

for _p in (ROOT / "src", ROOT / "src" / "faithfulness", ROOT / "src" / "grounding",
           ROOT / "src" / "ingestion"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@st.cache_resource(show_spinner="Loading the NLI verifier (once per session)...")
def load_verifier():
    from staged import StagedFaithfulness
    scorer = StagedFaithfulness.from_config(llm=None)
    scorer.verifier  # noqa: B018 -- force the model load now, not on first click
    return scorer


@st.cache_data(show_spinner=False)
def load_statements_for(dataset: str) -> dict:
    from staged import load_statements
    return {q: e.statements for q, e in load_statements(
        CKPT / f"statements_{dataset}_dev.jsonl").items()}


def _sentences(record: dict) -> list:
    from segmenter import evidence_sentences
    return evidence_sentences(record["retrieved"])


def _rebuild(sentences, drop: set) -> list:
    from perturb import _rebuild as rebuild
    return rebuild(sentences, drop)


def _targets(sentences, grounding_record: dict | None) -> dict:
    """position -> claim text, for sentences the grounding layer says supplied an operand."""
    if not grounding_record:
        return {}
    index = {(s.chunk_id, s.index): pos for pos, s in enumerate(sentences)}
    out = {}
    for claim in grounding_record["grounding"]["support"]:
        if not claim["supported"] or claim.get("chunk_id") is None:
            continue
        pos = index.get((claim["chunk_id"], claim.get("sentence_index")))
        if pos is not None:
            out.setdefault(pos, []).append(claim["claim"])
    return out


def _score_chart(live: dict, cached: dict | None, verifier_live: str, verifier_cached: str | None):
    """Two grouped bars per condition: live NLI vs recorded LLM (if any)."""
    conds = [c for c in ("control", "targeted", "random", "your selection") if c in live]
    fig = go.Figure()
    short_live, colour_live = verifier_style(verifier_live)
    fig.add_bar(name=f"{short_live} — live", x=conds, y=[live[c] for c in conds],
                marker_color=colour_live, text=[f"{live[c]:.2f}" for c in conds],
                textposition="outside", hovertemplate="%{x}: %{y:.3f}<extra>" + short_live + "</extra>")
    if cached:
        short_c, colour_c = verifier_style(verifier_cached or "llm")
        ys = [cached.get(c) for c in conds]
        fig.add_bar(name=f"{short_c} — recorded", x=conds,
                    y=[y if y is not None else 0 for y in ys], marker_color=colour_c,
                    text=[f"{y:.2f}" if y is not None else "—" for y in ys],
                    textposition="outside",
                    hovertemplate="%{x}: %{y:.3f}<extra>" + short_c + "</extra>")
    fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group", height=300,
                      margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15], title="faithfulness"),
                      legend=dict(orientation="h", y=1.12, x=0), font=dict(size=12))
    return fig


def render() -> None:
    st.markdown('<div class="ar-eyebrow">Audit playground — Contribution 2, hands on</div>',
                unsafe_allow_html=True)
    st.markdown(pill("live", "NLI verifier re-scores on every change") +
                pill("replay", "LLM verifier: recorded scores for the same three conditions"),
                unsafe_allow_html=True)
    dataset = sidebar_dataset()
    art = load_artifacts(dataset)
    if not art["b1"]:
        st.error(f"No B1 checkpoint for {dataset}.")
        st.stop()
    qa_id = pick_question(art, default_filter="audit-ready (grounded, scored > 0)")
    if qa_id is None:
        st.info("No question matches this filter.")
        return
    rec = art["b1"][qa_id]
    statements = load_statements_for(dataset).get(qa_id, [])
    if not statements:
        st.warning("The judge produced no statements for this answer, so there is nothing to "
                   "verify — pick another question. (These are the 'unscorable' rows.)")
        return

    # ---- the question and what RAGAS is checking ----
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Q:** {rec['question']}")
        st.markdown(f"**Answer:** `{rec['generated'].get('answer')}`  ·  gold `{rec['gold_answer']}`"
                    + (f"  ·  expression `{rec['generated']['answer_expression']}`"
                       if rec["generated"].get("answer_expression") else ""))
    with c2:
        st.markdown("**Statements RAGAS extracted from the answer** (fixed; the judge wrote "
                    "these once):")
        for s_ in statements:
            st.markdown(f"- {s_}")

    sentences = _sentences(rec)
    targets = _targets(sentences, art["grounding"].get(qa_id))
    n = len(sentences)

    # ---- controls ----
    # Checkbox widgets own their state under `{key}::{pos}`; the buttons write those states
    # directly (before the checkboxes render) so a click and a tick are the same thing.
    key = f"drop::{dataset}::{qa_id}"
    if key not in st.session_state:
        st.session_state[key] = True
        for pos in range(n):
            st.session_state[f"{key}::{pos}"] = True

    def _apply(drop_set: set) -> None:
        for pos in range(n):
            st.session_state[f"{key}::{pos}"] = pos not in drop_set

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Remove the operand sentences (targeted)", disabled=not targets,
                 help="The sentences the grounding layer says supplied the answer's figures"):
        _apply(set(targets))
    if b2.button(f"Remove {max(len(targets), 1)} random other sentence(s)"):
        pool = [i for i in range(n) if i not in targets]
        rng = random.Random(st.session_state.get("seed", 0))
        _apply(set(rng.sample(pool, min(len(pool), max(len(targets), 1)))))
        st.session_state["seed"] = st.session_state.get("seed", 0) + 1
    if b3.button("Reset (all evidence)"):
        _apply(set())
    b4.caption(f"{n} evidence sentences · {len(targets)} operand-supplying")

    # ---- the evidence, tick to remove ----
    st.markdown("**Evidence** — untick a sentence to remove it from the context. Highlighted "
                "rows supplied an operand of the answer.")
    drop = set()
    for pos, s_ in enumerate(sentences):
        label = s_.text[:180] + ("…" if len(s_.text) > 180 else "")
        tag = f"  ⟵ supplies {', '.join(targets[pos])}" if pos in targets else ""
        if not st.checkbox(f"`{s_.chunk_id}[{s_.index}]` {label}{tag}", key=f"{key}::{pos}"):
            drop.add(pos)

    # ---- live verification ----
    scorer = load_verifier()
    with st.spinner("Re-verifying with the NLI verifier..."):
        control = scorer.verify(statements, _rebuild(sentences, set()))
        current = scorer.verify(statements, _rebuild(sentences, drop))
        targeted = scorer.verify(statements, _rebuild(sentences, set(targets))) if targets else None

    live = {"control": control.score}
    if targeted is not None:
        live["targeted"] = targeted.score
    if drop and drop != set(targets):
        live["your selection"] = current.score
    # the recorded random arm for this question, if the NLI audit had one
    nli_pert = next((d["perturbation"].get(qa_id) for lbl, d in art["verifiers"].items()
                     if verifier_style(lbl)[0].startswith("NLI") and d["perturbation"].get(qa_id)), None)
    if nli_pert:
        live["random"] = nli_pert["scores"]["random"]
    cached, cached_label = None, None
    for lbl, d in art["verifiers"].items():
        p = d["perturbation"].get(qa_id)
        if p and lbl.startswith("llm"):
            cached, cached_label = p["scores"], lbl
            break

    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(_score_chart(live, cached, scorer.verifier_model, cached_label),
                        width="stretch")
        st.caption("control = all evidence · targeted = operand sentences removed · random = "
                   "equal number of other sentences removed (recorded seed-0 arm) · your "
                   "selection = what you ticked. Specificity = targeted drop − random drop.")
        if cached is None:
            st.caption("This question is not in the 80-question LLM-verifier audit subset, so "
                       "no recorded LLM scores exist for it; the NLI verifier is live for every "
                       "question.")
    with right:
        st.markdown(f"**Live verdicts — {verifier_style(scorer.verifier_model)[0]}, "
                    f"score {current.score:.2f}** ({len(drop)} sentence(s) removed)")
        for v in current.verdicts:
            icon = "✅" if v["supported"] else "❌"
            st.markdown(f"{icon} {v['statement']}  \n<span class='ar-source'>entailment "
                        f"{v.get('entailment', 0):.3f}</span>", unsafe_allow_html=True)
        if targeted is not None and nli_pert:
            spec = (control.score - targeted.score) - (control.score - nli_pert["scores"]["random"])
            st.markdown(f"**Specificity for this question: {spec:+.2f}**")
        if cached:
            ctrl, tgt, rnd = cached.get("control"), cached.get("targeted"), cached.get("random")
            if None not in (ctrl, tgt, rnd):
                st.markdown(f"**{verifier_style(cached_label)[0]} (recorded): "
                            f"{ctrl:.2f} → {tgt:.2f} targeted, {rnd:.2f} random — specificity "
                            f"{(ctrl - tgt) - (ctrl - rnd):+.2f}**")
    st.markdown('<div class="ar-source">Sources: checkpoints/statements_*.jsonl (statements) · '
                'checkpoints/grounding_*.jsonl (operand sentences) · '
                'checkpoints/perturbation_*_dev{,_LLMVER}.jsonl (recorded arms)</div>',
                unsafe_allow_html=True)
