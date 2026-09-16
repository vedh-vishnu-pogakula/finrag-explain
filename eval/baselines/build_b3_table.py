"""
The B1 / B2 / B3 comparison the brief promised -- built from results already on disk.

B3 is not a different generator. The frozen configuration (7B, 4-bit, program-of-thought,
three exemplars) is an experimental constant shared by all three baselines, so B3's answers
are B1's answers. What B3 adds is the explanation layer, and the question this table answers
is whether that layer is *worth anything*: a user who only trusts the answers B3 marks as
verified -- does that user end up holding more correct answers than one who trusts B2's flag,
or every answer?

Two tables:

  1. **What each baseline reports.** B1: retrieval + answer accuracy + citation precision.
     B2: the same plus a faithfulness number -- one number, which is how RAGAS is used in
     practice and which Month 5 showed is not interpretable without naming the verifier.
     B3: faithfulness across verifiers, groundedness against operand provenance, attribution
     lift, and perturbation specificity.

  2. **Selective accuracy.** Each baseline exposes a "verified" flag; report its coverage
     (fraction of answers flagged) and the accuracy among flagged vs unflagged answers.
       B1  -- no flag; every answer is trusted (coverage 1.0, accuracy = B1 accuracy).
       B2  -- faithfulness >= threshold under the named verifier.
       B3  -- operand provenance (deterministic) AND faithfulness >= threshold.
     A flag is useful if accuracy|verified is well above accuracy|unverified. A flag that
     cannot separate them is decoration, whatever its mean score.

Everything is joined by `qa_id` from the checkpoints; nothing is re-run. Declined answers
("insufficient evidence") count as unverified and incorrect for every baseline: abstaining is
the flag doing its job, not a free pass.

    python eval/baselines/build_b3_table.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval" / "results"
CKPT = RESULTS / "checkpoints"


def _load_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["qa_id"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def is_grounded(grounding_record: dict | None, threshold: float) -> bool:
    """B3's deterministic half. Numeric answers: every operand found in the evidence. Text
    answers (no numeric claims): groundedness >= threshold from the NLI text check."""
    if not grounding_record:
        return False
    g = grounding_record["grounding"]
    numeric = [s for s in g["support"] if s["kind"] == "numeric"]
    if numeric:
        return all(s["supported"] for s in numeric)
    return (g.get("groundedness") or 0.0) >= threshold


def selective_accuracy(flags: list[tuple[bool, bool]]) -> dict:
    """flags: (verified, correct) per question."""
    n = len(flags)
    ver = [c for v, c in flags if v]
    unv = [c for v, c in flags if not v]
    return {
        "n": n,
        "coverage": round(len(ver) / n, 4) if n else None,
        "accuracy_verified": round(sum(ver) / len(ver), 4) if ver else None,
        "accuracy_unverified": round(sum(unv) / len(unv), 4) if unv else None,
        "accuracy_overall": round(sum(c for _, c in flags) / n, 4) if n else None,
        "n_correct_verified": sum(ver), "n_correct_total": sum(c for _, c in flags),
    }


def _verifier_files(dataset: str) -> list[tuple[str, Path]]:
    out = []
    for path in sorted(glob.glob(str(CKPT / f"b2_{dataset}_dev*.jsonl"))):
        path = Path(path)
        suffix = path.name.replace(f"b2_{dataset}_dev", "").replace(".jsonl", "")
        if suffix.startswith("verdicts") or "SPOT" in suffix:
            continue
        cfg = _load_json(RESULTS / f"b2_{dataset}_dev{suffix}.json").get("config", {})
        model = cfg.get("verifier")
        model = model[0] if isinstance(model, list) and model else model
        label = f"{cfg.get('verifier_kind', 'nli')}:{str(model).split('/')[-1]}"
        out.append((label, path))
    return out


def build(dataset: str, threshold: float) -> dict:
    b1_sum = _load_json(RESULTS / f"b1_rag_{dataset}_dev.json")
    g_sum = _load_json(RESULTS / f"grounding_{dataset}_dev.json")
    attr = _load_json(RESULTS / f"attribution_{dataset}_dev_ranking.json") \
        or _load_json(RESULTS / f"attribution_{dataset}_dev_score.json")
    comparison = _load_json(RESULTS / "verifier_comparison.json").get(dataset, [])
    failures = _load_json(RESULTS / "failure_cases_summary.json").get("labels", {}).get(dataset, {})
    if not b1_sum:
        raise SystemExit(f"[b3] no B1 summary for {dataset}")

    answers = b1_sum["answers"]
    # Retrieval numbers come from the retrieval-only baseline, the file README cites. The RAG
    # run re-embeds and its MRR differs in the third decimal (tie order), so quoting both
    # would look like a discrepancy rather than the same retriever.
    retrieval = _load_json(RESULTS / f"b1_retrieval_{dataset}_dev.json") or b1_sum
    reported = {
        "B1": {
            "recall@5": retrieval["overall"]["recall@5"], "mrr": retrieval["overall"]["mrr"],
            "numeric_accuracy": answers["numeric_accuracy"],
            # FinQA has a single textual question; a span F1 over n=1 is noise, not a metric.
            "span_token_f1": answers["span_token_f1"] if answers["n_textual"] >= 10 else None,
            "citation_precision": b1_sum["citation_precision"],
        },
        "B2": {
            "faithfulness_by_verifier": {row["verifier"]: row["mean"] for row in comparison},
        },
        "B3": {
            "faithfulness_separation_by_verifier": {
                row["verifier"]: row["separation"] for row in comparison},
            "faithfulness_specificity_by_verifier": {
                row["verifier"]: row["specificity"] for row in comparison},
            "groundedness": g_sum.get("overall", {}).get("groundedness"),
            "grounded_hits_gold": g_sum.get("overall", {}).get("grounded_hits_gold"),
            "attribution_shapley_lift@0.2": attr.get("by_method", {}).get("shapley", {})
            .get("lift_over_random@0.2"),
            "attribution_occlusion_lift@0.2": attr.get("by_method", {}).get("occlusion", {})
            .get("lift_over_random@0.2"),
            "failure_cases_flagged": {v: s.get("any_label") for v, s in failures.items()},
        },
    }

    # ---- selective accuracy, per question -------------------------------------------------
    b1 = _load_jsonl(CKPT / f"b1_rag_{dataset}_dev.jsonl")
    grounding = _load_jsonl(CKPT / f"grounding_{dataset}_dev.jsonl")
    # Same definition run_b2.py uses for `correct` (numeric match, or exact match for spans,
    # and not declined) -- checked identical on all 500 questions, so the separation column
    # in verifier_comparison.json and this table agree on which answers are right.
    correct = {q: bool(r["answer_scores"].get("numeric_match") or
                       r["answer_scores"].get("exact_match")) and r["failure_mode"] != "declined"
               for q, r in b1.items()}

    selective = {"B1 (trust everything)": selective_accuracy([(True, c) for c in correct.values()])}
    grounded = {q: is_grounded(grounding.get(q), threshold) for q in b1}
    selective["B3 grounding only (no verifier)"] = selective_accuracy(
        [(grounded[q], correct[q]) for q in b1])
    for label, path in _verifier_files(dataset):
        b2 = _load_jsonl(path)
        faithful = {q: (b2.get(q, {}).get("faithfulness") or 0.0) >= threshold for q in b1}
        selective[f"B2  faithful [{label}]"] = selective_accuracy(
            [(faithful[q], correct[q]) for q in b1])
        selective[f"B3  grounded AND faithful [{label}]"] = selective_accuracy(
            [(grounded[q] and faithful[q], correct[q]) for q in b1])

    return {"reported": reported, "selective_accuracy": selective}


def _md(report: dict, threshold: float) -> str:
    lines = ["# B1 / B2 / B3 comparison", "",
             f"Generated by `eval/baselines/build_b3_table.py` (threshold {threshold}). "
             "All three baselines share the frozen generator; B2/B3 differ only in what they "
             "verify.", ""]
    for dataset, rep in report.items():
        r = rep["reported"]
        lines += [f"## {dataset.upper()}", "", "### What each baseline reports", "",
                  "| metric | B1 | B2 | B3 |", "|---|---|---|---|"]
        b1 = r["B1"]
        lines.append(f"| retrieval recall@5 / MRR | {b1['recall@5']:.3f} / {b1['mrr']:.3f} | same | same |")
        acc = f"{b1['numeric_accuracy']:.3f}"
        if b1["span_token_f1"] is not None:
            acc += f" numeric / span F1 {b1['span_token_f1']:.3f}"
        lines.append(f"| answer accuracy | {acc} | same | same |")
        lines.append(f"| citation precision | {b1['citation_precision']:.3f} | same | same |")
        for v, m in r["B2"]["faithfulness_by_verifier"].items():
            sep = r["B3"]["faithfulness_separation_by_verifier"].get(v)
            spec = r["B3"]["faithfulness_specificity_by_verifier"].get(v)
            sep_s = f"{sep:+.3f}" if sep is not None else "--"
            spec_s = f"{spec:+.3f}" if spec is not None else "pending"
            lines.append(f"| faithfulness [{v}] | -- | {m:.3f} | {m:.3f}; separation {sep_s}; "
                         f"specificity {spec_s} |")
        b3 = r["B3"]
        if b3["groundedness"] is not None:
            lines.append(f"| groundedness / grounds-to-gold | -- | -- | "
                         f"{b3['groundedness']:.3f} / {b3['grounded_hits_gold']:.3f} |")
        if b3["attribution_shapley_lift@0.2"] is not None:
            lines.append(f"| attribution lift over random @0.2 (Shapley / occlusion) | -- | -- | "
                         f"{b3['attribution_shapley_lift@0.2']:.3f} / "
                         f"{b3['attribution_occlusion_lift@0.2']:.3f} |")
        if b3["failure_cases_flagged"]:
            flagged = ", ".join(f"{v}: {n}" for v, n in b3["failure_cases_flagged"].items())
            lines.append(f"| verifier failure cases flagged | -- | -- | {flagged} |")
        lines += ["", "### Selective accuracy — trust only what the baseline marks verified", "",
                  "Correct = numeric match (or exact match for spans), declined answers count as "
                  "wrong. A flag earns its place only if acc. verified is well above acc. "
                  "unverified.", "",
                  "| baseline | coverage | acc. verified | acc. unverified | correct kept |",
                  "|---|---|---|---|---|"]
        for name, s in rep["selective_accuracy"].items():
            av = f"{s['accuracy_verified']:.3f}" if s["accuracy_verified"] is not None else "--"
            au = f"{s['accuracy_unverified']:.3f}" if s["accuracy_unverified"] is not None else "--"
            lines.append(f"| {name} | {s['coverage']:.3f} | {av} | {au} | "
                         f"{s['n_correct_verified']}/{s['n_correct_total']} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="B1/B2/B3 comparison from existing results")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="faithfulness / groundedness >= threshold counts as verified")
    ap.add_argument("--out", default=str(RESULTS / "b3_comparison"))
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else ["finqa", "tatqa"]
    report = {ds: build(ds, args.threshold) for ds in datasets}
    md = _md(report, args.threshold)
    print(md)
    Path(args.out + ".json").write_text(json.dumps(
        {"threshold": args.threshold, **report}, indent=2))
    Path(args.out + ".md").write_text(md)
    print(f"[b3] wrote {args.out}.json and .md")


if __name__ == "__main__":
    main()
