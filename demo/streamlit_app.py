"""
Explainable Financial RAG -- the demo.

What is live and what is replayed, stated up front because the distinction is the point:

  * **Retrieval and attribution run live** on the laptop. They are embedding-only (bge-small +
    one matmul per coalition), so a Shapley explanation of a query costs well under a second.
    Type any question against a document and both recompute.
  * **Generation, grounding, faithfulness and the perturbation audit are replayed** from the
    checkpoints in `eval/results/checkpoints/`. Generation is Qwen2.5-7B in 4-bit on Colab's
    T4 -- it does not fit on the laptop and the project has no budget for an API -- and the
    other three read its output. Faking a live 7B answer with a smaller model would show
    numbers that are not the paper's numbers, so the demo does not.

Every number on screen therefore traces to a file a reviewer can open. The 250 questions per
dataset are the paper's evaluation subsample, and the verifier table, grounding verdicts and
perturbation bars are the same rows `compare_verifiers.py` and `build_b3_table.py` aggregate.

    streamlit run demo/streamlit_app.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "src" / "ingestion", ROOT / "src" / "retrieval",
           ROOT / "src" / "attribution", ROOT / "src" / "grounding",
           ROOT / "eval" / "baselines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS = ROOT / "eval" / "results"
CKPT = RESULTS / "checkpoints"

st.set_page_config(page_title="Explainable Financial RAG", layout="wide")


# ---- replayed artifacts -------------------------------------------------------------------

def _jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["qa_id"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


@st.cache_data(show_spinner=False)
def load_artifacts(dataset: str) -> dict:
    """Everything the checkpoints know about this dataset's 250 questions, keyed by qa_id."""
    b1 = _jsonl(CKPT / f"b1_rag_{dataset}_dev.jsonl")
    grounding = _jsonl(CKPT / f"grounding_{dataset}_dev.jsonl")
    verifiers = {}
    for path in sorted(CKPT.glob(f"b2_{dataset}_dev*.jsonl")):
        suffix = path.name.replace(f"b2_{dataset}_dev", "").replace(".jsonl", "")
        if suffix.startswith("verdicts") or "SPOT" in suffix:
            continue
        cfg = {}
        summary = RESULTS / f"b2_{dataset}_dev{suffix}.json"
        if summary.exists():
            cfg = json.loads(summary.read_text()).get("config", {})
        model = cfg.get("verifier")
        model = model[0] if isinstance(model, list) and model else model
        label = f"{cfg.get('verifier_kind', 'nli')}: {str(model).split('/')[-1]}"
        verifiers[label] = {
            "scores": _jsonl(path),
            "perturbation": _jsonl(CKPT / f"perturbation_{dataset}_dev{suffix}.jsonl"),
        }
    failures = {}
    fpath = RESULTS / f"failure_cases_{dataset}_dev.jsonl"
    if fpath.exists():
        for line in open(fpath):
            if line.strip():
                r = json.loads(line)
                failures.setdefault(r["qa_id"], {})[r["verifier"]] = r["labels"]
    return {"b1": b1, "grounding": grounding, "verifiers": verifiers, "failures": failures}


# ---- live retrieval + attribution ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model and chunk index (once per session)...")
def load_retriever(dataset: str):
    """The same retriever the evaluation used: same embedder, same document-scoped index,
    same 250-question subsample, so a live retrieval here reproduces the checkpointed one."""
    from config_utils import get as cfg_get, load_config, resolve_path
    from embedder import Embedder
    from retriever import Retriever
    from run_b1_retrieval import get_or_build_index, index_dir_for, load_dataset, subsample

    cfg = load_config()
    raw = resolve_path(cfg_get(cfg, f"datasets.{dataset}.dev"))
    if not raw.exists():
        return None, f"{raw} not found -- run `bash scripts/download_data.sh` first."
    limit = int(cfg_get(cfg, "evaluation.subsample_size", 250))
    chunks, questions = load_dataset(dataset, raw, "dev")
    questions = subsample(questions, limit)
    needed = {q.doc_id for q in questions}
    chunks = [c for c in chunks if c.doc_id in needed]
    embedder = Embedder.from_config(cfg)
    index = get_or_build_index(chunks, embedder, index_dir_for(cfg, dataset, "dev", embedder,
                                                                limit), needed, False)
    retriever = Retriever(embedder=embedder, index=index,
                          top_k=int(cfg_get(cfg, "retrieval.top_k", 5)), scope="document")
    return retriever, None


@st.cache_data(show_spinner=False)
def explain(dataset: str, question: str, doc_id: str, method: str, top_k: int) -> dict:
    from explainer import explain_query

    retriever, err = load_retriever(dataset)
    if retriever is None:
        return {"error": err}
    hits = retriever.retrieve(question, doc_id=doc_id)
    exp = explain_query(retriever, question, doc_id=doc_id, method=method, mode="score",
                        top_k=top_k)
    return {"hits": [h.to_json() for h in hits], "explanation": exp.to_json()}


# ---- rendering helpers --------------------------------------------------------------------

def _highlight_units(units: list[dict], attributions: list[dict]) -> str:
    """The question with each unit shaded by its attribution weight."""
    weight = {a["index"]: a["weight"] for a in attributions}
    top = max((abs(w) for w in weight.values()), default=0.0) or 1.0
    spans = []
    for u in units:
        w = weight.get(u["index"], 0.0)
        alpha = min(abs(w) / top, 1.0) * 0.85
        color = f"rgba(46,125,50,{alpha:.2f})" if w >= 0 else f"rgba(198,40,40,{alpha:.2f})"
        spans.append(f'<span title="{u["kind"]}: {w:+.4f}" style="background:{color};'
                     f'padding:2px 4px;border-radius:3px;margin:1px">'
                     f'{html.escape(u["text"])}</span>')
    return " ".join(spans)


def _verdict_badge(rec: dict) -> str:
    mode = rec.get("failure_mode")
    if mode is None:
        return "✅ correct"
    return {"generation": "❌ wrong answer (gold evidence was retrieved)",
            "retrieval": "❌ wrong answer (retrieval missed the gold evidence)",
            "declined": "⚠️ declined -- model said the evidence was insufficient"}.get(
        mode, f"❌ {mode}")


# ---- page ---------------------------------------------------------------------------------

def main():
    st.title("Explainable Financial RAG")
    st.caption("Retrieval attribution + faithfulness-verified explanations for financial QA "
               "-- B.E. major project, CBIT Hyderabad")

    with st.sidebar:
        dataset = st.radio("Dataset", ["finqa", "tatqa"], horizontal=True)
        art = load_artifacts(dataset)
        if not art["b1"]:
            st.error(f"No B1 checkpoint for {dataset}. Copy "
                     f"`eval/results/checkpoints/b1_rag_{dataset}_dev.jsonl` from Drive.")
            st.stop()
        show = st.selectbox("Show", ["all questions", "correct", "wrong (generation)",
                                     "wrong (retrieval)", "declined",
                                     "verifier failure cases"])
        ids = sorted(art["b1"])
        if show == "correct":
            ids = [q for q in ids if art["b1"][q]["failure_mode"] is None]
        elif show.startswith("wrong (generation"):
            ids = [q for q in ids if art["b1"][q]["failure_mode"] == "generation"]
        elif show.startswith("wrong (retrieval"):
            ids = [q for q in ids if art["b1"][q]["failure_mode"] == "retrieval"]
        elif show == "declined":
            ids = [q for q in ids if art["b1"][q]["failure_mode"] == "declined"]
        elif show == "verifier failure cases":
            ids = [q for q in ids if any(art["failures"].get(q, {}).values())]
        st.caption(f"{len(ids)} of {len(art['b1'])} questions")
        qa_id = st.selectbox("Question", ids,
                             format_func=lambda q: art["b1"][q]["question"][:90])
        method = st.selectbox("Attribution method", ["shapley", "occlusion", "surrogate"])
        st.divider()
        st.markdown("**Live:** retrieval, attribution  \n**Replayed from checkpoints:** "
                    "generation (Qwen2.5-7B, Colab T4), grounding, faithfulness, perturbation")

    rec = art["b1"][qa_id]
    gen = rec["generated"]

    # ---- 1. question and answer ----
    st.subheader("1. Question and answer")
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.markdown(f"**Q:** {rec['question']}")
    c2.metric("Gold", str(rec["gold_answer"]))
    c3.metric("Predicted", str(gen.get("answer")))
    st.markdown(_verdict_badge(rec))
    if gen.get("answer_expression"):
        st.code(f"answer_expression = {gen['answer_expression']}", language="text")
        st.caption("Program-of-thought: the model named the arithmetic; "
                   "`src/generation/calculator.py` executed it.")
    st.caption(f"doc `{rec['doc_id']}` · gold evidence {rec['gold_chunk_ids']} · "
               f"cited {gen.get('cited_chunk_ids')} · prompt `{gen.get('prompt_version')}`")

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


if __name__ == "__main__":
    main()
