"""
Results dashboard -- the paper's figures, interactive, read from the same files.

Every chart here has a matplotlib twin in `eval/figures/make_figures.py`; the numbers come
from the identical JSON, so the demo and the paper cannot disagree. Hover any mark for the
exact value; the tables underneath are the WCAG-clean twin of each chart.
"""
from __future__ import annotations

import glob
import json
import statistics as st_
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import (AQUA, BLUE, CKPT, CRITICAL, GOOD, MUTED, ORANGE, PLOTLY_TEMPLATE, RESULTS,
                    VIOLET, _fmt, verifier_style)

DS = {"finqa": "FinQA", "tatqa": "TAT-QA"}
ORDER = ["NLI cross-encoder", "Phi-3.5-mini (3.8B LLM)", "Qwen2.5-7B (LLM)"]
SHORT = {"NLI cross-encoder": "NLI", "Phi-3.5-mini (3.8B LLM)": "Phi-3.5 (3.8B)", "Qwen2.5-7B (LLM)": "Qwen-7B"}


def _json(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _order_key(label: str) -> int:
    name = verifier_style(label)[0]
    return ORDER.index(name) if name in ORDER else 99


def _stars(p) -> str:
    if p is None or p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _layout(fig, height=330, **kw):
    fig.update_layout(template=PLOTLY_TEMPLATE, height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=12), legend=dict(orientation="h", y=-0.18, x=0), **kw)
    return fig


def _source(text: str) -> None:
    st.markdown(f'<div class="ar-source">Source: {text}</div>', unsafe_allow_html=True)


# ---- 1. master table ---------------------------------------------------------------------

def master_table() -> None:
    comp = _json("verifier_comparison.json")
    if not comp:
        return st.warning("run `python eval/baselines/compare_verifiers.py`")
    st.markdown("#### The master table — the same answers scored by each verifier")
    st.caption("Three validity axes per verifier. **separation** = mean(correct) − mean(wrong) · "
               "**specificity** = targeted-removal drop − random-removal drop · **provenance** = "
               "grounded − ungrounded against a deterministic reference. Stars: permutation tests.")
    axes = [("separation", "separation"), ("specificity", "specificity"), ("provenance", "provenance")]
    datasets = [d for d in ("finqa", "tatqa") if comp.get(d)]
    fig = make_subplots(rows=len(datasets), cols=3,
                        subplot_titles=[f"{DS[d]} — {a[1]}" for d in datasets for a in axes],
                        vertical_spacing=0.22, horizontal_spacing=0.08)
    for r_i, ds in enumerate(datasets, start=1):
        rows = sorted(comp[ds], key=lambda r: _order_key(r["verifier"]))
        names = [SHORT.get(verifier_style(r["verifier"])[0], verifier_style(r["verifier"])[0])
                 for r in rows]
        for c, (key, _) in enumerate(axes, start=1):
            vals = [r[key] if r[key] is not None else 0 for r in rows]
            fig.add_bar(x=names, y=vals, marker_color=[verifier_style(r["verifier"])[1] for r in rows],
                        text=[f"{r[key]:+.3f} {_stars(r.get(key + '_p'))}" if r[key] is not None
                              else "not run" for r in rows],
                        textposition="outside", showlegend=False, cliponaxis=False,
                        hovertemplate="%{x}<br>" + key + " %{y:+.3f}<extra></extra>",
                        row=r_i, col=c)
            fig.add_hline(y=0, line_color="#c3c2b7", line_width=1, row=r_i, col=c)
    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=10))
    st.plotly_chart(_layout(fig, height=280 * len(datasets) + 60), width="stretch")
    rows = [{"dataset": DS[d], "verifier": verifier_style(r["verifier"])[0], "n": r["n"],
             "mean": f"{r['mean']:.3f}", "separation": _fmt(r["separation"], r["separation_p"]),
             "specificity": _fmt(r["specificity"], r["specificity_p"])
             + (f" (n={r['n_perturbed']})" if r["n_perturbed"] else ""),
             "provenance": _fmt(r["provenance"], r["provenance_p"])}
            for d in datasets for r in sorted(comp[d], key=lambda r: _order_key(r["verifier"]))]
    st.dataframe(rows, width="stretch", hide_index=True)
    _source("eval/results/verifier_comparison.json — regenerate with compare_verifiers.py")


# ---- 2. correct vs wrong ----------------------------------------------------------------

def separation_chart() -> None:
    st.markdown("#### Does the score know a right answer from a wrong one?")
    fig = make_subplots(rows=1, cols=2, subplot_titles=[DS["finqa"], DS["tatqa"]], shared_yaxes=True)
    any_ = False
    for c, ds in enumerate(("finqa", "tatqa"), start=1):
        files = []
        for path in sorted(glob.glob(str(CKPT / f"b2_{ds}_dev*.jsonl"))):
            suffix = Path(path).name.replace(f"b2_{ds}_dev", "").replace(".jsonl", "")
            if suffix.startswith("verdicts") or "SPOT" in suffix or "SEED" in suffix:
                continue
            cfg = (_json(f"b2_{ds}_dev{suffix}.json") or {}).get("config", {})
            model = cfg.get("verifier")
            model = model[0] if isinstance(model, list) and model else model
            files.append((f"{cfg.get('verifier_kind', 'nli')}: {str(model).split('/')[-1]}", path))
        files.sort(key=lambda f: _order_key(f[0]))
        names, ok, bad = [], [], []
        for label, path in files:
            rows = [json.loads(l) for l in open(path) if l.strip()]
            rows = [r for r in rows if r.get("faithfulness") is not None]
            names.append(verifier_style(label)[0])
            ok.append(st_.mean(r["faithfulness"] for r in rows if r["correct"]))
            bad.append(st_.mean(r["faithfulness"] for r in rows if not r["correct"]))
        if names:
            any_ = True
            fig.add_bar(x=names, y=ok, name="correct answers", marker_color=GOOD, showlegend=c == 1,
                        text=[f"{v:.2f}" for v in ok], textposition="outside",
                        hovertemplate="%{x}<br>correct: %{y:.3f}<extra></extra>", row=1, col=c)
            fig.add_bar(x=names, y=bad, name="wrong answers", marker_color=CRITICAL, showlegend=c == 1,
                        text=[f"{v:.2f}" for v in bad], textposition="outside",
                        hovertemplate="%{x}<br>wrong: %{y:.3f}<extra></extra>", row=1, col=c)
    if not any_:
        return st.warning("no B2 checkpoints found")
    fig.update_yaxes(range=[0, 1.05], title_text="mean faithfulness", row=1, col=1)
    st.plotly_chart(_layout(fig, barmode="group"), width="stretch")
    _source("eval/results/checkpoints/b2_*_dev*.jsonl — per-question faithfulness split by B1 correctness")


# ---- 3. perturbation audit ---------------------------------------------------------------

def perturbation_chart() -> None:
    st.markdown("#### Perturbation audit — does the score react to WHICH evidence is removed?")
    fig = make_subplots(rows=1, cols=2, subplot_titles=[DS["finqa"], DS["tatqa"]], shared_yaxes=True)
    any_ = False
    for c, ds in enumerate(("finqa", "tatqa"), start=1):
        runs = []
        for suffix in ("", "_LLMVER", "_LLMVER-ALT", "_LLMVER-ALT2"):
            p = _json(f"perturbation_{ds}_dev{suffix}.json")
            if p:
                runs.append((p["config"]["verifier_model"], p["effects"]))
        runs.sort(key=lambda r: _order_key(r[0]))
        for verifier, e in runs:
            any_ = True
            short, colour = verifier_style(verifier)
            x = [f"{short}<br>n={e['n_paired']} · spec {e['specificity']:+.3f} {_stars(e['p_specificity'])}"]
            for cond, alpha, label in (("control", 1.0, "control (all evidence)"),
                                       ("targeted", 0.45, "operand sentence removed"),
                                       ("random", 0.7, "equal random sentences removed")):
                fig.add_bar(x=x, y=[e[cond]], name=label, marker_color=colour, opacity=alpha,
                            legendgroup=cond, showlegend=(c == 1 and verifier == runs[0][0]),
                            text=[f"{e[cond]:.2f}"], textposition="outside",
                            hovertemplate=f"{short}<br>{label}: " + "%{y:.3f}<extra></extra>",
                            row=1, col=c)
    if not any_:
        return st.warning("no perturbation results found")
    fig.update_yaxes(range=[0, 1.1], title_text="mean faithfulness", row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=9))
    st.plotly_chart(_layout(fig, height=380, barmode="group"), width="stretch")
    var = _json("perturbation_variance.json")
    if var:
        st.caption("Seed robustness (NLI verifier, random arm re-drawn): " + " · ".join(
            f"{DS[d]} specificity {v['specificity_mean']:+.3f} ± {v['specificity_sd']:.3f} over "
            f"{v['n_seeds']} seeds, deterministic arms byte-identical = "
            f"{v['control_and_targeted_identical_across_seeds']}" for d, v in var.items()))
    _source("eval/results/perturbation_*_dev{,_LLMVER}.json · perturbation_variance.json")


# ---- 4. selective accuracy ---------------------------------------------------------------

def selective_chart() -> None:
    b3 = _json("b3_comparison.json")
    if not b3:
        return st.warning("run `python eval/baselines/build_b3_table.py`")
    st.markdown("#### Selective accuracy — what a user who trusts only 'verified' answers keeps")
    st.caption("Each baseline exposes a *verified* flag. x = share of answers flagged, y = accuracy "
               "among them. Dashed line = trusting everything (B1). Circles = B2 (RAGAS faithful "
               "≥ 0.5), squares = provenance only, diamonds = B3 (provenance AND faithful).")
    fig = make_subplots(rows=1, cols=2, subplot_titles=[DS["finqa"], DS["tatqa"]], shared_yaxes=True)
    for c, ds in enumerate(("finqa", "tatqa"), start=1):
        if ds not in b3:
            continue
        sel = b3[ds]["selective_accuracy"]
        base = sel["B1 (trust everything)"]["accuracy_overall"]
        fig.add_hline(y=base, line_dash="dash", line_color=MUTED, row=1, col=c,
                      annotation_text=f"B1 trust everything {base:.2f}", annotation_position="top right",
                      annotation_font_size=10, annotation_font_color=MUTED)
        for name, s in sel.items():
            if s["accuracy_verified"] is None or name.startswith("B1"):
                continue
            if "[" in name:
                short, colour = verifier_style(name[name.find("[") + 1:name.find("]")])
            else:
                short, colour = "operand provenance only", VIOLET
            symbol = "circle" if name.startswith("B2") else ("square" if "grounding only" in name else "diamond")
            flag = "B2 faithful" if name.startswith("B2") else ("provenance only" if "grounding only" in name else "B3 grounded ∧ faithful")
            fig.add_scatter(x=[s["coverage"]], y=[s["accuracy_verified"]], mode="markers",
                            marker=dict(size=14, color=colour, symbol=symbol,
                                        line=dict(color="white", width=1)),
                            name=f"{flag} · {short}", showlegend=c == 1,
                            hovertemplate=(f"{flag}<br>{short}<br>coverage %{{x:.2f}}<br>"
                                           f"accuracy among verified %{{y:.3f}}<br>"
                                           f"correct kept {s['n_correct_verified']}/{s['n_correct_total']}"
                                           "<extra></extra>"),
                            row=1, col=c)
    fig.update_xaxes(range=[0, 1.05], title_text="coverage")
    fig.update_yaxes(range=[0.3, 0.85], title_text="accuracy among verified", row=1, col=1)
    st.plotly_chart(_layout(fig, height=420), width="stretch")
    _source("eval/results/b3_comparison.json — regenerate with build_b3_table.py")


# ---- 5. attribution ----------------------------------------------------------------------

def attribution_chart() -> None:
    files = [("FinQA · chunk score", "attribution_finqa_dev_score.json"),
             ("FinQA · ranking (RBO)", "attribution_finqa_dev_ranking.json"),
             ("TAT-QA · chunk score", "attribution_tatqa_dev_score.json")]
    data = [(lbl, _json(f)) for lbl, f in files if (RESULTS / f).exists()]
    if not data:
        return st.warning("no attribution results")
    st.markdown("#### Retrieval attribution (Contribution 1) — lift over a random-unit baseline")
    fig = go.Figure()
    for m, colour, label in (("occlusion", BLUE, "occlusion"), ("shapley", AQUA, "Shapley (RankingSHAP)"),
                             ("surrogate", ORANGE, "surrogate (Rank-LIME)")):
        ys = [d["by_method"][m]["lift_over_random@0.2"] for _, d in data]
        fig.add_bar(name=label, x=[f"{lbl} (n={d['n_questions']})" for lbl, d in data], y=ys,
                    marker_color=colour, text=[f"{y:.3f}" for y in ys], textposition="outside",
                    hovertemplate="%{x}<br>" + label + ": %{y:.3f}<extra></extra>")
    fig.update_yaxes(range=[0, 0.62], title_text="comprehensiveness lift @ top-20%")
    st.plotly_chart(_layout(fig, barmode="group"), width="stretch")
    rows = [{"file": lbl, "method": m, "lift @0.2": f"{v['lift_over_random@0.2']:.3f}",
             "norm. sufficiency @0.2": f"{v['norm_sufficiency@0.2']:+.3f}",
             "stopword mass": f"{v['mean_mass_by_kind'].get('stopword', 0):.3f}",
             "number mass": f"{v['mean_mass_by_kind'].get('number', 0):.3f}"}
            for lbl, d in data for m, v in d["by_method"].items()]
    st.dataframe(rows, width="stretch", hide_index=True)
    _source("eval/results/attribution_*_dev_{score,ranking}.json — regenerate with run_attribution.py")


# ---- 6. baseline + grounding -------------------------------------------------------------

def baseline_table() -> None:
    st.markdown("#### Baseline B1 and grounding, 250 questions per dataset")
    rows = []
    for ds in ("finqa", "tatqa"):
        b1, rt, g = _json(f"b1_rag_{ds}_dev.json"), _json(f"b1_retrieval_{ds}_dev.json"), _json(f"grounding_{ds}_dev.json")
        if not b1:
            continue
        ret = rt["overall"] if rt else b1["overall"]
        a = b1["answers"]
        rows.append({"dataset": DS[ds], "recall@5": f"{ret['recall@5']:.3f}", "MRR": f"{ret['mrr']:.3f}",
                     "numeric accuracy": f"{a['numeric_accuracy']:.3f}",
                     "span F1": f"{a['span_token_f1']:.3f}" if a["n_textual"] >= 10 else "—",
                     "citation precision": f"{b1['citation_precision']:.3f}",
                     "failures gen / retr / declined": f"{b1['failure_modes'].get('generation', 0)} / "
                     f"{b1['failure_modes'].get('retrieval', 0)} / {b1['failure_modes'].get('declined', 0)}",
                     "groundedness": f"{g['overall']['groundedness']:.3f}" if g else "—",
                     "grounds to gold": f"{g['overall']['grounded_hits_gold']:.3f}" if g else "—"})
    st.dataframe(rows, width="stretch", hide_index=True)
    abl7, abl1 = _json("ablation_finqa_dev_7b.json"), _json("ablation_finqa_dev.json")
    if abl7 and abl1:
        fig = go.Figure()
        arms = [r["arm"] for r in abl7["rows"]]
        for model, d, colour in (("Qwen2.5-1.5B", abl1, BLUE), ("Qwen2.5-7B", abl7, AQUA)):
            acc = {r["arm"]: r["numeric_accuracy"] for r in d["rows"]}
            fig.add_bar(name=model, x=arms, y=[acc[a] for a in arms], marker_color=colour,
                        text=[f"{acc[a]:.2f}" for a in arms], textposition="outside",
                        hovertemplate="%{x}<br>" + model + ": %{y:.2f}<extra></extra>")
        fig.update_yaxes(range=[0, 0.62], title_text="FinQA execution accuracy")
        st.plotly_chart(_layout(fig, height=300, barmode="group",
                                title=f"Generation ablation, {abl7['limit']} questions per arm"),
                        width="stretch")
    _source("eval/results/b1_rag_*_dev.json · b1_retrieval_*_dev.json · grounding_*_dev.json · ablation_finqa_dev{,_7b}.json")


# ---- 7. failure labels + agreement -------------------------------------------------------

def failure_and_agreement() -> None:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        s = _json("failure_cases_summary.json")
        if s:
            st.markdown("#### The labeled failure-case dataset")
            labels = [("false_negative", BLUE, "false negative"), ("false_positive", ORANGE, "false positive"),
                      ("insensitive", AQUA, "insensitive"), ("nonspecific", "#eda100", "nonspecific"),
                      ("unscorable", "#c3c2b7", "unscorable")]
            rows = [(d, v, c) for d, per in s["labels"].items()
                    for v, c in sorted(per.items(), key=lambda kv: _order_key(kv[0]))]
            y = [f"{DS[d]} · {verifier_style(v)[0]}" for d, v, _ in rows]
            fig = go.Figure()
            for key, colour, label in labels:
                fig.add_bar(name=label, y=y, x=[c[key] for _, _, c in rows], orientation="h",
                            marker_color=colour, hovertemplate="%{y}<br>" + label + ": %{x}<extra></extra>")
            fig.update_layout(barmode="stack", yaxis=dict(autorange="reversed"))
            fig.update_xaxes(title_text="labeled cases out of 250 per verifier")
            st.plotly_chart(_layout(fig, height=360), width="stretch")
            _source("eval/results/failure_cases_summary.json · rows in failure_cases_*_dev.jsonl")
    with c2:
        st.markdown("#### Do the two verifiers agree?")
        rows = []
        for ds in ("finqa", "tatqa"):
            r = _json(f"verifier_crosscheck_{ds}_dev.json")
            if r:
                a = r["agreement"]
                rows.append({"dataset": DS[ds], "questions": r["config"]["n_compared"],
                             "statements": a["n_statements"], "agreement": f"{a['verdict_agreement']:.3f}",
                             "Cohen's κ": f"{a['cohens_kappa']:+.4f}",
                             "LLM-only supported": a["llm_only_supported"],
                             "NLI-only supported": a["nli_only_supported"]})
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption("Statement-level agreement between the NLI cross-encoder and the 7B LLM "
                       "verifier on 50 questions per dataset. κ near 0 = chance.")
            _source("eval/results/verifier_crosscheck_*_dev.json")


def render() -> None:
    st.markdown('<div class="ar-eyebrow">Results — every figure in the paper, interactive</div>',
                unsafe_allow_html=True)
    st.caption("Hover any mark for the exact value. Each chart names the file it was read from; "
               "`python eval/figures/make_figures.py` draws the same data for the paper.")
    master_table()
    st.divider()
    separation_chart()
    st.divider()
    perturbation_chart()
    st.divider()
    selective_chart()
    st.divider()
    attribution_chart()
    st.divider()
    baseline_table()
    st.divider()
    failure_and_agreement()
