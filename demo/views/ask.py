"""Ask a question -- one question end to end, with retrieval and attribution recomputed live."""
from __future__ import annotations

import streamlit as st

from common import (_b3_verdict, _highlight_units, _verdict_badge, explain, load_artifacts,
                    pick_question, pill, sidebar_dataset)


def render() -> None:
    st.markdown('<div class="ar-eyebrow">Ask a question</div>', unsafe_allow_html=True)
    st.markdown(pill("live", "retrieval + attribution computed now") +
                pill("replay", "generation · grounding · faithfulness · perturbation from checkpoints"),
                unsafe_allow_html=True)
    dataset = sidebar_dataset()
    art = load_artifacts(dataset)
    if not art["b1"]:
        st.error(f"No B1 checkpoint for {dataset}.")
        st.stop()
    qa_id = pick_question(art)
    if qa_id is None:
        st.info("No question matches this filter.")
        return
    method = st.sidebar.selectbox("Attribution method", ["shapley", "occlusion", "surrogate"],
                                  key="method")
    render_question(dataset, qa_id, method, art)


def render_question(dataset: str, qa_id: str, method: str, art: dict) -> None:
    rec = art["b1"][qa_id]
    gen = rec["generated"]

    # ---- 1. question and answer ----
    st.subheader("1. Question and answer")
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.markdown(f"**Q:** {rec['question']}")
    c2.metric("Gold", str(rec["gold_answer"]))
    c3.metric("Predicted", str(gen.get("answer")))
    st.markdown(_verdict_badge(rec))
    verified, why, ref = _b3_verdict(qa_id, art)
    (st.success if verified else st.warning)(
        f"**B3 verdict: {why}**  \n"
        f"Verified = operand provenance (deterministic) AND faithfulness ≥ 0.5 under "
        f"`{ref}`. Across the 250 questions this flag raises accuracy among trusted answers "
        f"— see the Results overview tab.")
    if gen.get("answer_expression"):
        st.code(f"answer_expression = {gen['answer_expression']}", language="text")
        st.caption("Program-of-thought: the model named the arithmetic; "
                   "`src/generation/calculator.py` executed it.")
    m = rec.get("metrics", {})
    st.caption(f"doc `{rec['doc_id']}` · gold evidence {rec['gold_chunk_ids']} · "
               f"cited {gen.get('cited_chunk_ids')} · retrieval recall@5 {m.get('recall@5', 0):.2f}, "
               f"MRR {m.get('mrr', 0):.2f} · prompt `{gen.get('prompt_version')}`")

    # ---- 2. retrieval + attribution (live) ----
    st.subheader("2. Retrieval attribution — Contribution 1 (live)")
    custom = st.text_input("Try a different question against the same document "
                           "(retrieval + attribution only; no generation on the laptop)",
                           value="")
    question = custom.strip() or rec["question"]
    with st.spinner("Retrieving and attributing..."):
        live = explain(dataset, question, rec["doc_id"], method, top_k=3)
    if "error" in live:
        st.warning(live["error"])
    else:
        exp = live["explanation"]
        gold = set(rec["gold_chunk_ids"] if isinstance(rec["gold_chunk_ids"], list) else [])
        cited = set(gen.get("cited_chunk_ids") or [])
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Top-5 retrieved chunks** (● gold evidence, ★ cited by the model)")
            for h in live["hits"]:
                tag = ("● " if h["chunk_id"] in gold else "") + ("★ " if h["chunk_id"] in cited else "")
                with st.expander(f"#{h['rank'] + 1} {tag}{h['chunk_id']}  ·  score {h['score']:.4f}",
                                 expanded=h["rank"] == 0):
                    st.write(h["text"])
        with right:
            st.markdown(f"**Which query units drove each chunk's score** ({method})")
            for ch in exp["chunks"]:
                st.markdown(f"#{ch['rank'] + 1} `{ch['chunk_id']}` — base score {ch['base_score']:.4f}")
                st.markdown(_highlight_units(exp["units"], ch["attributions"]),
                            unsafe_allow_html=True)
                top = sorted(ch["attributions"], key=lambda a: -a["weight"])[:5]
                st.bar_chart({a["text"]: a["weight"] for a in top}, horizontal=True, height=160)
            stats = exp.get("stats", {})
            st.caption(f"{stats.get('n_units', len(exp['units']))} units · "
                       f"{stats.get('variants_requested', '?')} coalitions requested, "
                       f"{stats.get('variants_embedded', '?')} embedded (cache hit rate "
                       f"{stats.get('cache_hit_rate', '?')}) — one embedding pass per distinct "
                       f"coalition, one matmul for every chunk.")

    if custom.strip():
        st.info("Sections 3–5 describe the checkpointed answer to the *original* question; "
                "a custom question has no generated answer to ground or verify.")

    # ---- 3. grounding ----
    st.subheader("3. Evidence grounding — where did each figure come from?")
    g = art["grounding"].get(qa_id)
    if not g:
        st.warning("No grounding record for this question.")
    else:
        gg = g["grounding"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Groundedness", f"{gg['groundedness']:.2f}")
        m2.metric("Claims", gg["n_claims"])
        m3.metric("Grounds to gold evidence", "yes" if g["agreement"]["grounded_hits_gold"] else "no")
        rows = []
        for s in gg["support"]:
            rows.append({"claim": s["claim"], "kind": s["kind"],
                         "supported": "✅" if s["supported"] else "❌",
                         "found in": f"{s.get('chunk_id')}[{s.get('sentence_index')}]"
                         if s["supported"] else "—",
                         "sentence": (s.get("sentence") or "")[:140]})
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption("Numeric claims are the operands of `answer_expression`, looked up "
                   "deterministically (no model). Text claims use DeBERTa-MNLI entailment.")

    # ---- 4. faithfulness across verifiers ----
    st.subheader("4. Faithfulness — the same answer under each verifier")
    st.caption("RAGAS decomposes the answer into statements (judge LLM, once), then a verifier "
               "checks each statement against the evidence. The paper's finding: which "
               "verifier you pick changes the verdict.")
    vcols = st.columns(max(len(art["verifiers"]), 1))
    for col, (label, data) in zip(vcols, art["verifiers"].items()):
        v = data["scores"].get(qa_id)
        with col:
            st.markdown(f"**{label}**")
            if not v or v.get("faithfulness") is None:
                st.write("unscorable (no statements)")
                continue
            st.metric("faithfulness", f"{v['faithfulness']:.2f}")
            for verdict in v.get("verdicts", []):
                icon = "✅" if verdict["supported"] else "❌"
                st.markdown(f"{icon} {verdict['statement']}")
                if verdict.get("reason"):
                    st.caption(f"reason: {verdict['reason'][:220]}")
                elif verdict.get("entailment") is not None:
                    st.caption(f"entailment {verdict['entailment']:.3f}")
            labels = art["failures"].get(qa_id, {}).get(label.replace(": ", ":"), [])
            if labels:
                st.error("failure labels: " + ", ".join(labels))

    # ---- 5. perturbation audit ----
    st.subheader("5. Perturbation audit — Contribution 2")
    st.caption("Remove the sentence grounding says supplied the operands (targeted) vs the same "
               "number of random sentences. A valid metric drops more under targeted removal.")
    any_pert = False
    pcols = st.columns(max(len(art["verifiers"]), 1))
    for col, (label, data) in zip(pcols, art["verifiers"].items()):
        p = data["perturbation"].get(qa_id)
        with col:
            st.markdown(f"**{label}**")
            if not p:
                st.write("not in this verifier's perturbation run")
                continue
            any_pert = True
            s = p["scores"]
            st.bar_chart({k: s[k] for k in ("control", "targeted", "random")
                          if s.get(k) is not None}, height=180)
            spec = (s["control"] - s["targeted"]) - (s["control"] - s["random"])
            st.caption(f"specificity = targeted drop − random drop = **{spec:+.2f}**")
            st.markdown("Removed (targeted):")
            for r in p["removed_targeted"]:
                st.markdown(f"- `{r['chunk_id']}[{r['sentence_index']}]` {r['text'][:120]}")
    if not any_pert:
        st.write("This question had no grounded operand sentence to remove, so it is not in "
                 "the audit.")

    st.divider()
    st.caption("Sources: `eval/results/checkpoints/{b1_rag,grounding,b2_*,perturbation_*}_"
               f"{dataset}_dev.jsonl`, `eval/results/failure_cases_{dataset}_dev.jsonl`. "
               "Aggregates: `eval/results/verifier_comparison.json`, `b3_comparison.md`.")
