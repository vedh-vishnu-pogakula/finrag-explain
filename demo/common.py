"""
AuditRAG demo -- shared loaders, helpers and theme.

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


def _operands_grounded(g: dict | None, threshold: float = 0.5) -> bool:
    """B3's deterministic half, same rule as eval/baselines/build_b3_table.py::is_grounded."""
    if not g:
        return False
    gg = g["grounding"]
    numeric = [s for s in gg["support"] if s["kind"] == "numeric"]
    if numeric:
        return all(s["supported"] for s in numeric)
    return (gg.get("groundedness") or 0.0) >= threshold


def _reference_verifier(verifiers: dict) -> str | None:
    """The verifier B3's flag uses: the 7B LLM (the one that passed both audit axes)."""
    for label in verifiers:
        if label.startswith("llm") and "Qwen" in label:
            return label
    return next(iter(verifiers), None)


def _b3_verdict(qa_id: str, art: dict, threshold: float = 0.5) -> tuple[bool, str, str | None]:
    grounded = _operands_grounded(art["grounding"].get(qa_id), threshold)
    ref = _reference_verifier(art["verifiers"])
    score = art["verifiers"][ref]["scores"].get(qa_id, {}).get("faithfulness") if ref else None
    faithful = score is not None and score >= threshold
    if grounded and faithful:
        return True, "VERIFIED — every operand found in the evidence, and the reference " \
                     f"verifier scored it faithful ({score:.2f})", ref
    reasons = []
    if not grounded:
        reasons.append("an operand was not found in the retrieved evidence")
    if score is None:
        reasons.append("no statements to verify")
    elif not faithful:
        reasons.append(f"reference verifier scored it {score:.2f} < {threshold}")
    return False, "NOT VERIFIED — " + "; ".join(reasons), ref


def _stars(p) -> str:
    if p is None or p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _fmt(v, p=None) -> str:
    return "—" if v is None else f"{v:+.3f}{_stars(p)}"




# ---- branding + theme --------------------------------------------------------------------

APP_NAME = "AuditRAG"
TAGLINE = "Explainable financial RAG with retrieval attribution and verifier-validated faithfulness"
PAPER_TITLE = "Auditing the Auditor: Verifier Choice and the Validity of Faithfulness Metrics in Financial RAG"

# Same palette as paper/figures (validated for colour-vision separation on adjacent pairs).
BLUE, AQUA, ORANGE, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eb6834", "#eda100", "#4a3aa7"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK, INK2, MUTED, HAIRLINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PLOTLY_TEMPLATE = "plotly_white"


def verifier_style(label: str) -> tuple[str, str]:
    """(short name, colour) fixed per verifier family -- identical to the paper figures."""
    low = label.lower()
    if low.startswith("nli") or "cross-encoder" in low or "deberta" in low:
        return "NLI cross-encoder", BLUE
    if "phi" in low:
        return "Phi-3.5-mini (3.8B LLM)", ORANGE
    if "qwen" in low:
        return "Qwen2.5-7B (LLM)", AQUA
    return label, VIOLET


def inject_css() -> None:
    st.markdown("""
<style>
@import url('data:text/css,');
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --hair:#e1e0d9;
        --blue:#2a78d6; --aqua:#1baf7a; --orange:#eb6834; --surface:#fcfcfb; }
h1, h2, h3 { letter-spacing: -0.01em; }
.ar-eyebrow { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
              color: var(--muted); font-weight: 600; }
.ar-hero-title { font-size: 2.6rem; font-weight: 700; line-height: 1.05; margin: 0.2rem 0 0.4rem; }
.ar-hero-sub { font-size: 1.05rem; color: var(--ink2); max-width: 62ch; }
.ar-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 1rem 0; }
.ar-tile { border: 1px solid var(--hair); border-radius: 10px; padding: 14px 16px; background: var(--surface); }
.ar-tile .k { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.ar-tile .v { font-size: 1.9rem; font-weight: 700; line-height: 1.1; margin: 4px 0 2px; }
.ar-tile .d { font-size: 0.82rem; color: var(--ink2); }
.ar-pill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.75rem;
           font-weight: 600; border: 1px solid var(--hair); margin-right: 6px; }
.ar-pill.live { color: #0a6b2a; background: #e9f7ee; border-color: #bfe6cc; }
.ar-pill.replay { color: #5b4a1a; background: #fbf3df; border-color: #eedcae; }
.ar-step { border-left: 3px solid var(--hair); padding: 4px 0 4px 14px; margin: 10px 0; }
.ar-step.blue { border-color: var(--blue); } .ar-step.aqua { border-color: var(--aqua); }
.ar-step.orange { border-color: var(--orange); }
.ar-source { font-size: 0.74rem; color: var(--muted); }
@media (prefers-reduced-motion: reduce) { .ar-anim * { animation: none !important; } }
</style>
""", unsafe_allow_html=True)


def pill(kind: str, text: str) -> str:
    return f'<span class="ar-pill {kind}">{text}</span>'


def tile(label: str, value: str, detail: str, colour: str = INK) -> str:
    return (f'<div class="ar-tile"><div class="k">{label}</div>'
            f'<div class="v" style="color:{colour}">{value}</div><div class="d">{detail}</div></div>')


def sidebar_dataset() -> str:
    """Dataset picker shared by the question-level pages; remembered across pages."""
    return st.sidebar.radio("Dataset", ["finqa", "tatqa"], horizontal=True, key="dataset")


FILTERS = ["all questions", "correct", "wrong (generation)", "wrong (retrieval)", "declined",
           "verifier failure cases", "audit-ready (grounded, scored > 0)"]


def _audit_ready(art: dict) -> set:
    """Questions where the NLI audit had a targeted arm and a non-zero control score -- the
    ones where removing evidence can visibly move the score."""
    for label, data in art["verifiers"].items():
        if verifier_style(label)[0].startswith("NLI"):
            return {q for q, p in data["perturbation"].items()
                    if (p["scores"].get("control") or 0) > 0}
    return set()


def pick_question(art: dict, default_filter: str = "all questions") -> str | None:
    """Sidebar question picker with the outcome filters; remembered across pages."""
    if "show" not in st.session_state:
        st.session_state["show"] = default_filter
    show = st.sidebar.selectbox("Show", FILTERS, key="show")
    ids = sorted(art["b1"])
    if show.startswith("audit-ready"):
        ready = _audit_ready(art)
        ids = [q for q in ids if q in ready]
    elif show == "correct":
        ids = [q for q in ids if art["b1"][q]["failure_mode"] is None]
    elif show.startswith("wrong (generation"):
        ids = [q for q in ids if art["b1"][q]["failure_mode"] == "generation"]
    elif show.startswith("wrong (retrieval"):
        ids = [q for q in ids if art["b1"][q]["failure_mode"] == "retrieval"]
    elif show == "declined":
        ids = [q for q in ids if art["b1"][q]["failure_mode"] == "declined"]
    elif show == "verifier failure cases":
        ids = [q for q in ids if any(art["failures"].get(q, {}).values())]
    st.sidebar.caption(f"{len(ids)} of {len(art['b1'])} questions")
    if not ids:
        return None
    return st.sidebar.selectbox("Question", ids, key="qa_id",
                                format_func=lambda q: art["b1"][q]["question"][:90])
