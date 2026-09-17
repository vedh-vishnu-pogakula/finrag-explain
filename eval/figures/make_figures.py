"""
Paper figures -- every one read from a file in `eval/results/`, none hand-typed.

    python eval/figures/make_figures.py            # -> paper/figures/fig??_*.{png,pdf}

Rules this script enforces on itself, because a figure that disagrees with its data is
worse than no figure:

  * No literal numbers. Every bar, point and error bar comes from a result file; the
    source path is printed under each figure as its caption.
  * A figure whose source file is missing is skipped with a message, not drawn from memory.
  * Verifier colors are fixed per verifier (NLI = blue, Phi = orange, Qwen = aqua, Falcon =
    yellow) so the same verifier looks the same in every figure, including ones added later.
  * Significance stars are recomputed from the p-values in the files, with the same
    thresholds as `compare_verifiers.py`.

Regenerate after any result file changes; the Falcon rows (cell 10g) appear automatically.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval" / "results"
CKPT = RESULTS / "checkpoints"
OUT = ROOT / "paper" / "figures"

# Categorical palette in fixed slot order (validated for adjacent-pair CVD separation).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GOOD, CRITICAL = "#0ca30c", "#d03b3b"            # status colours: correct / wrong only
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DATASET_NAME = {"finqa": "FinQA", "tatqa": "TAT-QA"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


# ---- helpers ------------------------------------------------------------------------------

def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _jsonl(path: Path) -> list:
    return [json.loads(l) for l in open(path) if l.strip()] if path.exists() else []


def _stars(p) -> str:
    if p is None or p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def verifier_style(label: str) -> tuple[str, str]:
    """(short name, colour) fixed per verifier family."""
    low = label.lower()
    if low.startswith("nli") or "cross-encoder" in low or "deberta" in low:
        return "NLI cross-encoder", BLUE
    if "phi" in low:
        return "LLM Phi-3.5-mini (3.8B)", ORANGE
    if "qwen" in low:
        return "LLM Qwen2.5-7B", AQUA
    if "falcon" in low:
        return "LLM Falcon3-7B", YELLOW
    return label, VIOLET


VERIFIER_ORDER = ["NLI cross-encoder", "LLM Phi-3.5-mini (3.8B)", "LLM Qwen2.5-7B",
                  "LLM Falcon3-7B"]


def _order_key(label: str) -> int:
    name = verifier_style(label)[0]
    return VERIFIER_ORDER.index(name) if name in VERIFIER_ORDER else 99


def _order(labels: list[str]) -> list[str]:
    """Same verifier order in every figure: NLI, Phi, Qwen, Falcon."""
    return sorted(labels, key=_order_key)


def _tick(label: str, extra: str = "") -> str:
    """Two-line tick label per verifier, short enough not to collide."""
    name = verifier_style(label)[0]
    short = {"NLI cross-encoder": "NLI\ncross-encoder",
             "LLM Phi-3.5-mini (3.8B)": "Phi-3.5-mini\n(3.8B LLM)",
             "LLM Qwen2.5-7B": "Qwen2.5\n(7B LLM)",
             "LLM Falcon3-7B": "Falcon3\n(7B LLM)"}.get(name, name)
    return short + (f"\n{extra}" if extra else "")


def _finish(fig, name: str, source: str, caption: str = "", suptitle: str | None = None,
            bottom: float = 0.05) -> None:
    top = 1.0
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, color=INK, x=0.01, ha="left", y=0.99)
        top = 0.95
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.text(0.01, 0.005, f"Source: {source}" + (f"  —  {caption}" if caption else ""),
             fontsize=6.5, color=MUTED, ha="left", va="bottom", transform=fig.transFigure)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.png/.pdf")


def _bar_labels(ax, bars, fmt="{:.2f}", dy=0.01, color=INK2):
    for b in bars:
        h = b.get_height()
        if h != h:
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + (dy if h >= 0 else -dy), fmt.format(h),
                ha="center", va="bottom" if h >= 0 else "top", fontsize=7, color=color)


def _verifier_files(dataset: str) -> list[tuple[str, Path]]:
    out = []
    for path in sorted(glob.glob(str(CKPT / f"b2_{dataset}_dev*.jsonl"))):
        path = Path(path)
        suffix = path.name.replace(f"b2_{dataset}_dev", "").replace(".jsonl", "")
        if suffix.startswith("verdicts") or "SPOT" in suffix or "SEED" in suffix:
            continue
        cfg = (_json(RESULTS / f"b2_{dataset}_dev{suffix}.json") or {}).get("config", {})
        model = cfg.get("verifier")
        model = model[0] if isinstance(model, list) and model else model
        out.append((f"{cfg.get('verifier_kind', 'nli')}: {str(model).split('/')[-1]}", path))
    return out


# ---- figures ------------------------------------------------------------------------------

def fig01_ablation():
    files = {"Qwen2.5-1.5B": RESULTS / "ablation_finqa_dev.json",
             "Qwen2.5-7B": RESULTS / "ablation_finqa_dev_7b.json"}
    data = {k: _json(p) for k, p in files.items()}
    if not all(data.values()):
        return print("  skip fig01: ablation files missing")
    arms = [r["arm"] for r in data["Qwen2.5-7B"]["rows"]]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    w, x = 0.38, np.arange(len(arms))
    for i, (model, colour) in enumerate([("Qwen2.5-1.5B", BLUE), ("Qwen2.5-7B", AQUA)]):
        acc = {r["arm"]: r["numeric_accuracy"] for r in data[model]["rows"]}
        bars = ax.bar(x + (i - 0.5) * w, [acc[a] for a in arms], w * 0.94, color=colour,
                      label=model)
        _bar_labels(ax, bars)
    ax.set_xticks(x, [a.replace("direct", "direct answer").replace("pot", "program-of-thought")
                      .replace("-0shot", "\n0 exemplars").replace("-3shot", "\n3 exemplars")
                      for a in arms], fontsize=7.5)
    ax.set_ylabel("FinQA execution accuracy")
    ax.set_ylim(0, max(r["numeric_accuracy"] for m in data.values() for r in m["rows"]) * 1.25)
    ax.set_title("Generation ablation: prompt form × exemplars, at two model sizes", fontsize=9,
                 color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8)
    n = data["Qwen2.5-7B"]["limit"]
    _finish(fig, "fig01_generation_ablation", "eval/results/ablation_finqa_dev{,_7b}.json",
            f"{n} questions per arm; same ordering at both sizes")


def fig02_retrieval():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    drawn = False
    for ax, ds in zip(axes, ("finqa", "tatqa")):
        r = _json(RESULTS / f"b1_retrieval_{ds}_dev.json")
        if not r:
            continue
        drawn = True
        ks = r["ks"]
        for key, colour, label in (("recall@", BLUE, "recall@k"), ("hit@", AQUA, "hit@k"),
                                   ("full_recall@", ORANGE, "all gold retrieved@k")):
            ys = [r["overall"][f"{key}{k}"] for k in ks]
            ax.plot(ks, ys, marker="o", ms=4, lw=2, color=colour, label=label)
            ax.annotate(f"{ys[-1]:.2f}", (ks[-1], ys[-1]), textcoords="offset points",
                        xytext=(4, -3), fontsize=7, color=INK2)
        ax.set_xticks(ks)
        ax.set_xlabel("k")
        ax.set_ylim(0.3, 1.02)
        ax.set_title(f"{DATASET_NAME[ds]} (n={r['overall']['n_questions']})", fontsize=9,
                     color=INK, loc="left")
    if not drawn:
        return print("  skip fig02: retrieval files missing")
    axes[0].set_ylabel("proportion")
    axes[0].legend(fontsize=7.5, loc="lower right")
    _finish(fig, "fig02_retrieval_recall", "eval/results/b1_retrieval_{finqa,tatqa}_dev.json",
            suptitle="Retrieval is not the bottleneck: bge-small, per-document scope")


def fig03_failure_modes():
    rows = {ds: _json(RESULTS / f"b1_rag_{ds}_dev.json") for ds in ("finqa", "tatqa")}
    if not any(rows.values()):
        return print("  skip fig03")
    order = [("None", "correct", GOOD), ("generation", "wrong: generation / arithmetic", CRITICAL),
             ("retrieval", "wrong: retrieval missed gold", ORANGE),
             ("declined", "declined (insufficient evidence)", MUTED)]
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for key, label, colour in order:
        vals = np.array([rows[ds]["failure_modes"].get(key, 0) / rows[ds]["answers"]["n_questions"]
                         if rows[ds] else 0 for ds in rows])
        bars = ax.barh(y, vals, left=left, color=colour, label=label, height=0.55,
                       edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            if v > 0.06:
                ax.text(b.get_x() + v / 2, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=7.5, color="white")
        left += vals
    ax.set_yticks(y, [f"{DATASET_NAME[ds]} (n={rows[ds]['answers']['n_questions']})"
                      for ds in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of questions")
    fig.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.06))
    ax.set_title("Where B1 loses: generation, not retrieval", fontsize=9, color=INK, loc="left")
    _finish(fig, "fig03_failure_modes", "eval/results/b1_rag_{finqa,tatqa}_dev.json",
            "failure_modes", bottom=0.3)


def fig04_attribution_lift():
    files = [("FinQA · chunk score", RESULTS / "attribution_finqa_dev_score.json"),
             ("FinQA · ranking (RBO)", RESULTS / "attribution_finqa_dev_ranking.json"),
             ("TAT-QA · chunk score", RESULTS / "attribution_tatqa_dev_score.json")]
    data = [(lbl, _json(p)) for lbl, p in files if p.exists()]
    if not data:
        return print("  skip fig04")
    methods = [("occlusion", BLUE, "occlusion"), ("shapley", AQUA, "Shapley (RankingSHAP)"),
               ("surrogate", ORANGE, "surrogate (Rank-LIME)")]
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    x, w = np.arange(len(data)), 0.26
    for i, (m, colour, label) in enumerate(methods):
        vals = [d["by_method"][m]["lift_over_random@0.2"] for _, d in data]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.94, color=colour, label=label)
        _bar_labels(ax, bars, "{:.3f}")
    ax.set_xticks(x, [f"{lbl}\n(n={d['n_questions']})" for lbl, d in data])
    ax.set_ylabel("comprehensiveness lift\nover random units @ top-20%")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.32))
    ax.set_title("Retrieval attribution: every estimator beats random; Shapley leads on ranking",
                 fontsize=9, color=INK, loc="left")
    _finish(fig, "fig04_attribution_lift", "eval/results/attribution_*_dev_{score,ranking}.json",
            "lift_over_random@0.2")


def fig05_attribution_mass():
    d = _json(RESULTS / "attribution_finqa_dev_score.json")
    if not d:
        return print("  skip fig05")
    kinds = [("number", BLUE), ("entity", AQUA), ("term", "#c3c2b7"), ("stopword", ORANGE)]
    methods = ["occlusion", "shapley", "surrogate"]
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    for kind, colour in kinds:
        vals = np.array([d["by_method"][m]["mean_mass_by_kind"].get(kind, 0) for m in methods])
        bars = ax.barh(y, vals, left=left, color=colour, label=kind, height=0.55,
                       edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            if v > 0.05:
                ax.text(b.get_x() + v / 2, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if colour != "#c3c2b7" else INK)
            elif kind == "stopword":            # the headline number; too thin to label inside
                ax.text(b.get_x() + v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.1%}",
                        ha="left", va="center", fontsize=7.5, color=INK2)
        left += vals
    ax.set_yticks(y, methods)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("share of attribution mass by query-unit kind")
    fig.legend(fontsize=7.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.06))
    ax.set_title(f"Where the mass goes (FinQA, chunk-score mode, n={d['n_questions']})",
                 fontsize=9, color=INK, loc="left")
    _finish(fig, "fig05_attribution_mass_by_kind", "eval/results/attribution_finqa_dev_score.json",
            "mean_mass_by_kind", bottom=0.25)


def fig06_grounding():
    data = {ds: _json(RESULTS / f"grounding_{ds}_dev.json") for ds in ("finqa", "tatqa")}
    if not any(data.values()):
        return print("  skip fig06")
    modes = [("None", "correct"), ("generation", "wrong\n(generation)"),
             ("retrieval", "wrong\n(retrieval)"), ("declined", "declined")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, ds in zip(axes, data):
        d = data[ds]
        if not d:
            continue
        x, w = np.arange(len(modes)), 0.38
        for i, (key, colour, label) in enumerate([("groundedness", BLUE, "groundedness"),
                                                  ("grounded_hits_gold", AQUA,
                                                   "grounds to gold evidence")]):
            vals = [d["by_failure_mode"].get(k, {}).get(key, np.nan) for k, _ in modes]
            bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.94, color=colour, label=label)
            _bar_labels(ax, bars)
        ax.set_xticks(x, [f"{lbl}\n(n={d['by_failure_mode'].get(k, {}).get('n', 0)})"
                          for k, lbl in modes], fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.set_title(DATASET_NAME[ds], fontsize=9, color=INK, loc="left")
    axes[0].legend(fontsize=7.5, loc="upper right")
    _finish(fig, "fig06_grounding_by_outcome", "eval/results/grounding_{finqa,tatqa}_dev.json",
            "by_failure_mode",
            suptitle="Evidence grounding by outcome (operand provenance + DeBERTa-MNLI)")


def fig07_master():
    comp = _json(RESULTS / "verifier_comparison.json")
    if not comp:
        return print("  skip fig07")
    cols = [("separation", "separation\nmean(correct) − mean(wrong)"),
            ("specificity", "specificity\ntargeted drop − random drop"),
            ("provenance", "provenance\ngrounded − ungrounded")]
    datasets = [ds for ds in ("finqa", "tatqa") if comp.get(ds)]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(7.4, 2.2 * len(datasets)),
                             squeeze=False)
    for r, ds in enumerate(datasets):
        rows = comp[ds]
        names = [verifier_style(x["verifier"]) for x in rows]
        for c, (key, title) in enumerate(cols):
            ax = axes[r][c]
            vals = [x[key] if x[key] is not None else np.nan for x in rows]
            bars = ax.bar(range(len(rows)), vals, 0.6, color=[n[1] for n in names])
            for b, x in zip(bars, rows):
                v = x[key]
                if v is None:
                    ax.text(b.get_x() + b.get_width() / 2, 0.01, "not\nrun", ha="center",
                            va="bottom", fontsize=6.5, color=MUTED)
                    continue
                s = _stars(x.get(f"{key}_p"))
                ax.text(b.get_x() + b.get_width() / 2, v + (0.015 if v >= 0 else -0.015),
                        f"{v:+.3f}\n{s}", ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=6.5, color=INK2)
            ax.axhline(0, color="#c3c2b7", lw=0.8)
            ax.set_xticks(range(len(rows)), [_tick(x["verifier"]) for x in rows], fontsize=6.5)
            lo = min([v for v in vals if v == v] + [0])
            hi = max([v for v in vals if v == v] + [0])
            ax.set_ylim(lo - 0.12 if lo < 0 else -0.04, hi + 0.18)
            if r == 0:
                ax.set_title(title, fontsize=8, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(f"{DATASET_NAME[ds]} (n={rows[0]['n']})", fontsize=8, color=INK)
    _finish(fig, "fig07_verifier_master", "eval/results/verifier_comparison.json",
            "* p<0.05  ** p<0.01  *** p<0.001 (permutation tests, 20k resamples); "
            "specificity n = questions with a grounded operand sentence",
            suptitle="Auditing the auditor: the same answers scored by each verifier")


def fig08_separation():
    datasets = ("finqa", "tatqa")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    drawn = False
    for ax, ds in zip(axes, datasets):
        files = _verifier_files(ds)
        if not files:
            continue
        drawn = True
        files = sorted(files, key=lambda f: _order_key(f[0]))
        x, w = np.arange(len(files)), 0.36
        names = []
        for i, (label, path) in enumerate(files):
            rows = [r for r in _jsonl(path) if r.get("faithfulness") is not None]
            ok = [r["faithfulness"] for r in rows if r["correct"]]
            bad = [r["faithfulness"] for r in rows if not r["correct"]]
            short, colour = verifier_style(label)
            names.append(_tick(label, f"n={len(ok)}/{len(bad)}"))
            b1 = ax.bar(i - w / 2, st.mean(ok), w * 0.94, color=GOOD,
                        label="correct answers" if i == 0 else None)
            b2 = ax.bar(i + w / 2, st.mean(bad), w * 0.94, color=CRITICAL,
                        label="wrong answers" if i == 0 else None)
            _bar_labels(ax, list(b1) + list(b2))
            ax.text(i, max(st.mean(ok), st.mean(bad)) + 0.08,
                    f"Δ {st.mean(ok) - st.mean(bad):+.2f}", ha="center", fontsize=7,
                    color=colour, fontweight="bold")
        ax.set_xticks(x, names, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_NAME[ds], fontsize=9, color=INK, loc="left")
    if not drawn:
        return print("  skip fig08")
    axes[0].set_ylabel("mean faithfulness")
    axes[0].legend(fontsize=7.5, loc="upper left")
    _finish(fig, "fig08_faithfulness_correct_vs_wrong", "eval/results/checkpoints/b2_*_dev*.jsonl",
            "faithfulness per question, split by B1 correctness; n = correct/wrong",
            suptitle="Does the score know a right answer from a wrong one?")


def fig09_restatement():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    drawn = False
    for ax, ds in zip(axes, ("finqa", "tatqa")):
        rows = _jsonl(RESULTS / f"failure_cases_{ds}_dev.jsonl")
        if not rows:
            continue
        drawn = True
        verifiers = _order(sorted({r["verifier"] for r in rows}))
        x, w, names = np.arange(len(verifiers)), 0.36, []
        for i, v in enumerate(verifiers):
            sub = [r for r in rows if r["verifier"] == v and r["faithfulness"] is not None
                   and r["restates_figure"] is not None]
            copied = [r["faithfulness"] for r in sub if r["restates_figure"]]
            computed = [r["faithfulness"] for r in sub if not r["restates_figure"]]
            short, colour = verifier_style(v)
            names.append(_tick(v, f"n={len(copied)}/{len(computed)}"))
            b1 = ax.bar(i - w / 2, st.mean(copied), w * 0.94, color=colour,
                        label="answer restates a figure in the evidence" if i == 0 else None)
            b2 = ax.bar(i + w / 2, st.mean(computed), w * 0.94, color=colour, alpha=0.4,
                        label="answer is computed (value not in evidence)" if i == 0 else None)
            _bar_labels(ax, list(b1) + list(b2))
        ax.set_xticks(x, names, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_NAME[ds], fontsize=9, color=INK, loc="left")
    if not drawn:
        return print("  skip fig09")
    axes[0].set_ylabel("mean faithfulness")
    axes[0].legend(fontsize=7, loc="upper left")
    _finish(fig, "fig09_restatement_mechanism", "eval/results/failure_cases_{finqa,tatqa}_dev.jsonl",
            "restates_figure = answer value appears numerically in retrieved evidence; "
            "n = restated/computed",
            suptitle="The mechanism: verifiers reward restatement over computation")


def fig10_perturbation():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    drawn = False
    for ax, ds in zip(axes, ("finqa", "tatqa")):
        runs = []
        for suffix in ("", "_LLMVER", "_LLMVER-ALT", "_LLMVER-ALT2"):
            p = _json(RESULTS / f"perturbation_{ds}_dev{suffix}.json")
            if p:
                runs.append((p["config"]["verifier_model"], p))
        if not runs:
            continue
        drawn = True
        runs = sorted(runs, key=lambda r: _order_key(r[0]))
        x, w = np.arange(len(runs)), 0.25
        names = []
        for i, (verifier, p) in enumerate(runs):
            e = p["effects"]
            short, colour = verifier_style(verifier)
            names.append(_tick(verifier, f"n={e['n_paired']}\nspecificity {e['specificity']:+.3f}"
                                         f" {_stars(e['p_specificity'])}"))
            for j, (cond, alpha, label) in enumerate([("control", 1.0, "control (all evidence)"),
                                                      ("targeted", 0.45, "remove operand sentence"),
                                                      ("random", 0.7, "remove equal random sentences")]):
                b = ax.bar(i + (j - 1) * w, e[cond], w * 0.94, color=colour, alpha=alpha,
                           label=label if i == 0 else None,
                           hatch="//" if cond == "targeted" else None, edgecolor="white")
                _bar_labels(ax, b)
        ax.set_xticks(x, names, fontsize=6.8)
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_NAME[ds], fontsize=9, color=INK, loc="left")
    if not drawn:
        return print("  skip fig10")
    axes[0].set_ylabel("mean faithfulness")
    axes[0].legend(fontsize=6.8, loc="upper left")
    _finish(fig, "fig10_perturbation_audit", "eval/results/perturbation_{finqa,tatqa}_dev{,_LLMVER}.json",
            "effects; hatched = operand-supplying sentence removed",
            suptitle="Perturbation audit: does the score react to WHICH evidence is removed?")


def fig11_seed_variance():
    v = _json(RESULTS / "perturbation_variance.json")
    if not v:
        return print("  skip fig11")
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    for i, (ds, colour) in enumerate([("finqa", BLUE), ("tatqa", AQUA)]):
        if ds not in v:
            continue
        seeds = sorted(v[ds]["per_seed"], key=int)
        ys = [v[ds]["per_seed"][s]["specificity"] for s in seeds]
        xs = np.arange(len(seeds)) + (i - 0.5) * 0.18
        ax.scatter(xs, ys, s=28, color=colour, zorder=3,
                   label=f"{DATASET_NAME[ds]}: {v[ds]['specificity_mean']:+.3f} ± "
                         f"{v[ds]['specificity_sd']:.3f}")
        ax.errorbar([len(seeds) + (i - 0.5) * 0.18], [v[ds]["specificity_mean"]],
                    yerr=[v[ds]["specificity_sd"]], fmt="D", ms=5, color=colour, capsize=3,
                    zorder=3)
        for xx, s, y in zip(xs, seeds, ys):
            p = v[ds]["per_seed"][s]["p"]
            ax.text(xx, y + 0.012, _stars(p), ha="center", fontsize=6.5, color=colour)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    n = max(len(v[ds]["per_seed"]) for ds in v)
    ax.set_xticks(list(range(n)) + [n], [f"seed {s}" for s in range(n)] + ["mean ± sd"])
    ax.set_ylabel("specificity (NLI verifier)")
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("Only the random arm is stochastic; the sign never flips", fontsize=9,
                 color=INK, loc="left")
    _finish(fig, "fig11_specificity_across_seeds", "eval/results/perturbation_variance.json",
            "control and targeted arms byte-identical across seeds")


def fig12_selective_accuracy():
    b3 = _json(RESULTS / "b3_comparison.json")
    if not b3:
        return print("  skip fig12")
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), sharey=True)
    seen_verifiers = {}
    for ax, ds in zip(axes, ("finqa", "tatqa")):
        if ds not in b3:
            continue
        sel = b3[ds]["selective_accuracy"]
        base = sel["B1 (trust everything)"]["accuracy_overall"]
        ax.axhline(base, color=MUTED, lw=1, ls=(0, (3, 2)))
        ax.text(1.03, base + 0.008, f"B1: trust everything ({base:.2f})", ha="right",
                fontsize=6.5, color=MUTED)
        for name, sacc in sel.items():
            if sacc["accuracy_verified"] is None or name.startswith("B1"):
                continue
            if "[" in name:
                vlabel = name[name.find("[") + 1:name.find("]")]
                short, colour = verifier_style(vlabel)
                seen_verifiers[short] = colour
            else:
                short, colour = "operand provenance only", VIOLET
                seen_verifiers[short] = colour
            marker = "o" if name.startswith("B2") else ("s" if "grounding only" in name else "D")
            ax.scatter(sacc["coverage"], sacc["accuracy_verified"], s=42, marker=marker,
                       color=colour, zorder=3, edgecolor="white", linewidth=0.8)
            if name.startswith("B3") and "Qwen" in name:       # the headline point only
                ax.annotate(f"B3 (Qwen): {sacc['accuracy_verified']:.2f} at coverage "
                            f"{sacc['coverage']:.2f}", (sacc["coverage"], sacc["accuracy_verified"]),
                            textcoords="offset points", xytext=(8, 6), ha="left", fontsize=6.5,
                            color=INK2)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0.3, 0.85)
        ax.set_title(DATASET_NAME[ds], fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("accuracy among verified answers")
    for ax in axes:
        ax.set_xlabel("coverage (share of answers marked verified)")
    shape_handles = [Line2D([], [], marker="o", ls="", color=INK2, label="B2: faithful ≥ 0.5"),
                     Line2D([], [], marker="s", ls="", color=INK2, label="grounding only"),
                     Line2D([], [], marker="D", ls="", color=INK2,
                            label="B3: grounded AND faithful")]
    colour_handles = [Line2D([], [], marker="o", ls="", color=c, label=v.replace("LLM ", ""))
                      for v, c in sorted(seen_verifiers.items(),
                                         key=lambda kv: _order_key(kv[0]) if kv[0] in VERIFIER_ORDER
                                         else 50)]
    fig.legend(handles=shape_handles + colour_handles, fontsize=6.5, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, 0.04), columnspacing=1.2, handletextpad=0.4)
    _finish(fig, "fig12_selective_accuracy", "eval/results/b3_comparison.json",
            "selective_accuracy; threshold 0.5", bottom=0.2,
            suptitle="Selective accuracy: what a user who trusts only 'verified' answers keeps")


def fig13_failure_labels():
    s = _json(RESULTS / "failure_cases_summary.json")
    if not s:
        return print("  skip fig13")
    labels = [("false_negative", BLUE, "false negative (correct + grounded, scored unfaithful)"),
              ("false_positive", ORANGE, "false positive (wrong / ungrounded, scored faithful)"),
              ("insensitive", AQUA, "insensitive (targeted removal: no drop)"),
              ("nonspecific", YELLOW, "nonspecific (targeted drop ≤ random drop)"),
              ("unscorable", "#c3c2b7", "unscorable (no statements)")]
    rows = [(ds, v, c) for ds, per in s["labels"].items()
            for v, c in sorted(per.items(), key=lambda kv: _order_key(kv[0]))]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.4))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for key, colour, label in labels:
        vals = np.array([c[key] for _, _, c in rows])
        bars = ax.barh(y, vals, left=left, color=colour, label=label, height=0.6,
                       edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            if v >= 8:
                ax.text(b.get_x() + v / 2, b.get_y() + b.get_height() / 2, str(v), ha="center",
                        va="center", fontsize=7, color="white" if colour != "#c3c2b7" else INK)
        left += vals
    ax.set_yticks(y, [f"{DATASET_NAME[ds]} · {verifier_style(v)[0].replace('LLM ', '')}"
                      for ds, v, _ in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"labeled cases out of n={rows[0][2]['n']} questions per verifier")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax.set_title("The labeled failure-case dataset (Contribution 2 deliverable)", fontsize=9,
                 color=INK, loc="left")
    _finish(fig, "fig13_failure_case_labels", "eval/results/failure_cases_summary.json",
            "perturbation labels only where that verifier has a perturbation run")


def fig14_crosscheck():
    """Per-question verdicts are almost all 0 or 1 (one statement per answer), so a scatter
    would be four corner blobs. A 2x2 agreement count is the honest form."""
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.9))
    drawn = False
    for ax, ds in zip(axes, ("finqa", "tatqa")):
        r = _json(RESULTS / f"verifier_crosscheck_{ds}_dev.json")
        if not r:
            continue
        drawn = True
        pts = r["per_question"]
        grid = np.zeros((2, 2), dtype=int)          # rows: LLM (1, 0); cols: NLI (0, 1)
        for p in pts:
            nli = int(p["nli_score"] >= 0.5)
            llm = int(p["llm_score"] >= 0.5)
            grid[1 - llm, nli] += 1
        ax.imshow(grid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "seq", ["#cde2fb", "#0d366b"]), vmin=0, vmax=max(grid.max(), 1))
        for i in range(2):
            for j in range(2):
                v = grid[i, j]
                ax.text(j, i, f"{v}\n({v / len(pts):.0%})", ha="center", va="center", fontsize=8,
                        color="white" if v > grid.max() * 0.55 else INK)
        ax.set_xticks([0, 1], ["NLI: unfaithful", "NLI: faithful"], fontsize=7)
        ax.set_yticks([0, 1], ["LLM:\nfaithful", "LLM:\nunfaithful"], fontsize=7)
        ax.grid(False)
        agree = (grid[1, 0] + grid[0, 1]) / len(pts)
        a = r["agreement"]
        ax.set_title(f"{DATASET_NAME[ds]} (n={len(pts)}): agree on {agree:.0%} of questions\n"
                     f"statement-level Cohen's κ = {a['cohens_kappa']}", fontsize=8, color=INK,
                     loc="left")
    if not drawn:
        return print("  skip fig14")
    _finish(fig, "fig14_verifier_crosscheck", "eval/results/verifier_crosscheck_{finqa,tatqa}_dev.json",
            "per_question scores binarised at 0.5; κ from statement-level verdicts",
            suptitle="Two verifiers RAGAS treats as interchangeable agree at chance")


FIGURES = [fig01_ablation, fig02_retrieval, fig03_failure_modes, fig04_attribution_lift,
           fig05_attribution_mass, fig06_grounding, fig07_master, fig08_separation,
           fig09_restatement, fig10_perturbation, fig11_seed_variance,
           fig12_selective_accuracy, fig13_failure_labels, fig14_crosscheck]


def main():
    ap = argparse.ArgumentParser(description="Regenerate every paper figure from eval/results")
    ap.add_argument("--only", default=None, help="substring of a figure function name")
    args = ap.parse_args()
    for fn in FIGURES:
        if args.only and args.only not in fn.__name__:
            continue
        print(f"[figures] {fn.__name__}")
        fn()


if __name__ == "__main__":
    main()
