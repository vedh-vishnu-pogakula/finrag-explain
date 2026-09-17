"""
The labeled failure-case dataset -- Contribution 2's stated deliverable.

Every faithfulness verdict this project produced is joined, per question and per verifier,
against three things it should agree with and demonstrably often does not:

  * the answer's **correctness** (B1, scored against FinQA's `exe_ans` / TAT-QA's gold),
  * its **operand provenance** (the grounding layer: are the figures the answer consumed
    provably in the retrieved evidence? deterministic, no model involved),
  * its **perturbation response** (control / targeted / random removal of evidence).

Each question gets zero or more labels. A question with no label is a case where the verifier
behaved as a faithfulness metric should. The labels are mechanical -- every one is a
comparison between numbers already on disk -- so the dataset is reproducible from the result
files with no model call, and a reviewer can re-derive any row.

  false_negative   answer correct AND every operand provably in the evidence, yet the verifier
                   scored it unfaithful. The metric's error, not the model's: nothing in the
                   answer is unsupported. (Mechanism: NLI cannot verify arithmetic.)
  false_positive   answer wrong with the gold evidence retrieved, or an operand missing from
                   the evidence, yet the verifier scored it faithful.
  insensitive      removing the sentence that supplies the answer's operands did not lower the
                   score at all, although there was a score to lower.
  nonspecific      removing that sentence lowered the score no more than removing an equal
                   number of random sentences -- the verifier reacts to how much evidence is
                   left, not which.
  unscorable       the judge produced no statements, so the verifier returned nothing.

`restates_figure` is recorded alongside as the mechanism flag: does the answer's value appear
verbatim in the evidence? A verifier that can only check restatement scores copied figures
as faithful and computed ones as unfaithful, which is exactly the false_positive /
false_negative pair above.

Output: `eval/results/failure_cases_<dataset>_dev.jsonl` (one row per question x verifier,
labels included) and `failure_cases_summary.json` (counts). Both are versioned -- they are the
deliverable.

    python eval/baselines/export_failure_cases.py
    python eval/baselines/export_failure_cases.py --dataset finqa --threshold 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval" / "results"
CKPT = RESULTS / "checkpoints"

LABELS = ("false_negative", "false_positive", "insensitive", "nonspecific", "unscorable")

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["qa_id"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


def _numeric_value(answer: str):
    """The bare magnitude of a numeric answer, or None for a text answer."""
    if answer is None:
        return None
    m = _NUM_RE.search(str(answer).replace("$", ""))
    if not m:
        return None
    try:
        return abs(float(m.group().replace(",", "")))
    except ValueError:
        return None


def restates_figure(answer: str, evidence_texts) -> bool | None:
    """Does the answer's value appear as a figure in the evidence? None for text answers.

    Compared numerically, not as strings, so `14`, `14.0` and `14.00` match and `1,400`
    matches `1400`. Sign is ignored: `-14` restates a `14` in a table."""
    value = _numeric_value(answer)
    if value is None:
        return None
    for text in evidence_texts:
        for tok in _NUM_RE.findall(text):
            try:
                if abs(abs(float(tok.replace(",", ""))) - value) < 1e-6:
                    return True
            except ValueError:
                continue
    return False


def operands_grounded(grounding: dict | None) -> bool | None:
    """True if every numeric claim was found in the evidence; None if there were none.

    Text claims are excluded on purpose: their support is an NLI judgement, and the point of
    this reference is that it involves no model."""
    if not grounding:
        return None
    numeric = [s for s in grounding["grounding"]["support"] if s["kind"] == "numeric"]
    if not numeric:
        return None
    return all(s["supported"] for s in numeric)


def label_case(*, faithfulness, correct, failure_mode, grounded, scores, threshold) -> list:
    """Every label the row earns. Pure so it can be tested without files."""
    labels = []
    if faithfulness is None:
        return ["unscorable"]
    faithful = faithfulness >= threshold
    # A retrieval failure is not the verifier's fault: the evidence really does not support
    # the answer, so "unfaithful" is the right verdict. Only generation failures -- gold
    # evidence retrieved, answer still wrong -- are the verifier's to catch.
    if not faithful and correct and grounded is True:
        labels.append("false_negative")
    if faithful and ((not correct and failure_mode == "generation") or grounded is False):
        labels.append("false_positive")
    if scores and all(scores.get(c) is not None for c in ("control", "targeted", "random")):
        control, targeted, random_ = scores["control"], scores["targeted"], scores["random"]
        if control > 0:
            if control - targeted <= 0:
                labels.append("insensitive")
            elif (control - targeted) <= (control - random_):
                labels.append("nonspecific")
    return labels


def _verifier_label(dataset: str, suffix: str) -> str:
    summary = RESULTS / f"b2_{dataset}_dev{suffix}.json"
    if summary.exists():
        cfg = json.loads(summary.read_text()).get("config", {})
        model = cfg.get("verifier")
        model = model[0] if isinstance(model, list) and model else model
        return f"{cfg.get('verifier_kind', 'nli')}:{str(model).split('/')[-1]}"
    return suffix.lstrip("_") or "nli-default"


def export(dataset: str, threshold: float) -> tuple[list, dict]:
    b1 = _load(CKPT / f"b1_rag_{dataset}_dev.jsonl")
    grounding = _load(CKPT / f"grounding_{dataset}_dev.jsonl")
    if not b1:
        raise SystemExit(f"[failures] no B1 records for {dataset} -- run run_b1_rag.py first")

    rows, summary = [], {}
    for path in sorted(glob.glob(str(CKPT / f"b2_{dataset}_dev*.jsonl"))):
        path = Path(path)
        suffix = path.name.replace(f"b2_{dataset}_dev", "").replace(".jsonl", "")
        if suffix.startswith("verdicts") or "SPOT" in suffix:
            continue
        verifier = _verifier_label(dataset, suffix)
        b2 = _load(path)
        pert = _load(CKPT / f"perturbation_{dataset}_dev{suffix}.jsonl")

        counts = Counter()
        for qa_id, rec in b1.items():
            verdict = b2.get(qa_id)
            if verdict is None:
                continue
            g = grounding.get(qa_id)
            p = pert.get(qa_id)
            evidence = [hit["text"] for hit in rec["retrieved"]]
            answer = rec["generated"].get("answer")
            grounded = operands_grounded(g)
            labels = label_case(
                faithfulness=verdict.get("faithfulness"), correct=bool(rec.get("correct",
                                                                            verdict["correct"])),
                failure_mode=rec.get("failure_mode"), grounded=grounded,
                scores=p["scores"] if p else None, threshold=threshold)
            counts.update(labels)
            counts["n"] += 1
            counts["n_perturbed"] += bool(p)
            rows.append({
                "qa_id": qa_id, "dataset": dataset, "verifier": verifier,
                "question": rec["question"], "gold_answer": rec["gold_answer"],
                "predicted": answer, "answer_expression": rec["generated"].get("answer_expression"),
                "correct": bool(verdict["correct"]), "failure_mode": rec.get("failure_mode"),
                "faithfulness": verdict.get("faithfulness"),
                "statements": [v["statement"] for v in verdict.get("verdicts", [])],
                "verdicts": [bool(v["supported"]) for v in verdict.get("verdicts", [])],
                "operands_grounded": grounded,
                "restates_figure": restates_figure(answer, evidence),
                "perturbation": p["scores"] if p else None,
                "removed_targeted": [s["text"] for s in p["removed_targeted"]] if p else None,
                "labels": labels,
            })
        n = counts["n"]
        summary[verifier] = {
            "n": n, "n_perturbed": counts["n_perturbed"],
            **{lab: counts[lab] for lab in LABELS},
            **{f"{lab}_rate": round(counts[lab] / n, 4) if n else None for lab in LABELS},
            "any_label": sum(1 for r in rows if r["verifier"] == verifier and r["labels"]),
        }
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="Export the labeled faithfulness failure cases")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="faithfulness >= threshold counts as 'faithful' (default 0.5)")
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else ["finqa", "tatqa"]
    report = {"threshold": args.threshold, "labels": {}}
    for dataset in datasets:
        rows, summary = export(dataset, args.threshold)
        out = Path(args.out_dir) / f"failure_cases_{dataset}_dev.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        report["labels"][dataset] = summary

        print(f"\n=== {dataset.upper()}  (faithful = score >= {args.threshold}) ===")
        print(f"{'verifier':36s} {'n':>4s} {'FN':>5s} {'FP':>5s} {'insens':>7s} "
              f"{'nonspec':>8s} {'unscor':>7s} {'any':>5s}")
        print("-" * 84)
        for verifier, s in summary.items():
            print(f"{verifier:36s} {s['n']:4d} {s['false_negative']:5d} {s['false_positive']:5d} "
                  f"{s['insensitive']:7d} {s['nonspecific']:8d} {s['unscorable']:7d} "
                  f"{s['any_label']:5d}")
        print(f"[failures] wrote {out} ({len(rows)} rows)")

    summary_path = Path(args.out_dir) / "failure_cases_summary.json"
    summary_path.write_text(json.dumps(report, indent=2))
    print("\n  FN = correct + operands grounded, scored unfaithful     FP = wrong (gold retrieved)"
          " or operand missing, scored faithful")
    print("  insens = targeted removal did not lower the score      nonspec = targeted drop <= "
          "random drop\n  (insens/nonspec only where a perturbation run exists for that verifier)")
    print(f"[failures] wrote {summary_path}")


if __name__ == "__main__":
    main()
