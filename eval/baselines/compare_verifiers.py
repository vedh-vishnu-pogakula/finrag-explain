"""
The paper's master table: what each verifier reports, and whether any of it means anything.

One command regenerates every headline number from the result files, so the table in the paper
is never hand-copied. Hand-copying is how a table ends up disagreeing with the run that
produced it, and a table that disagrees with its own data is the fastest way to lose a viva.

Four columns, because a faithfulness verifier can pass some and fail others -- and the whole
Month 5/6 finding is that they do:

  mean            what the verifier reports. On its own it says nothing: a verifier can report
                  0.80 and be useless.
  separation      correct answers minus incorrect. Does the score track whether the answer is
                  right? A metric that cannot do this is not measuring answer quality.
  specificity     targeted-removal drop minus random-removal drop. Does the score respond to
                  *which* evidence was removed rather than *how much*?
  provenance      grounded minus ungrounded, against operand provenance -- a deterministic
                  reference with no model in it.

`separation` and `specificity` are independent, and the paper's methodological claim rests on
that: a verifier can be evidence-specific (passes the perturbation audit) while failing to
track correctness, because the sentence supplying the operands is also the sentence supplying
the lexical overlap.

    python eval/baselines/compare_verifiers.py
    python eval/baselines/compare_verifiers.py --dataset finqa
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval" / "results"
CKPT = RESULTS / "checkpoints"


def _permutation_p(a, b, trials: int = 20000) -> float:
    """Two-sample permutation test. Non-parametric on purpose: faithfulness scores are bounded
    in [0,1] and heavily clustered at the ends, so a t-test's normality assumption does not
    hold."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    observed = abs(st.mean(a) - st.mean(b))
    pool, n, rng, hits = list(a) + list(b), len(a), random.Random(0), 0
    for _ in range(trials):
        rng.shuffle(pool)
        hits += abs(st.mean(pool[:n]) - st.mean(pool[n:])) >= observed
    return hits / trials


def _paired_p(deltas, trials: int = 20000) -> float:
    if not deltas:
        return float("nan")
    observed, rng, hits = abs(st.mean(deltas)), random.Random(0), 0
    for _ in range(trials):
        hits += abs(st.mean([d if rng.random() < 0.5 else -d for d in deltas])) >= observed
    return hits / trials


def _fmt(value, p=None) -> str:
    if value is None or value != value:
        return "  --  "
    star = ""
    if p is not None and p == p:
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{value:+.3f}{star}"


def _grounding(dataset):
    path = CKPT / f"grounding_{dataset}_dev.jsonl"
    if not path.exists():
        return {}
    return {json.loads(l)["qa_id"]: json.loads(l) for l in open(path) if l.strip()}


def analyse(dataset: str, b2_path: Path, label: str) -> dict:
    rows = [json.loads(l) for l in open(b2_path) if l.strip()]
    scored = [r for r in rows if r.get("faithfulness") is not None]
    ok = [r["faithfulness"] for r in scored if r["correct"]]
    bad = [r["faithfulness"] for r in scored if not r["correct"]]

    ground = _grounding(dataset)
    full, part = [], []
    for r in scored:
        g = ground.get(r["qa_id"])
        if not g:
            continue
        numeric = [s for s in g["grounding"]["support"] if s["kind"] == "numeric"]
        if not numeric:
            continue
        (full if all(s["supported"] for s in numeric) else part).append(r["faithfulness"])

    out = {
        "verifier": label,
        "n": len(scored),
        "mean": round(st.mean([r["faithfulness"] for r in scored]), 4) if scored else None,
        "separation": round(st.mean(ok) - st.mean(bad), 4) if ok and bad else None,
        "separation_p": _permutation_p(ok, bad) if ok and bad else float("nan"),
        "provenance": round(st.mean(full) - st.mean(part), 4) if full and part else None,
        "provenance_p": _permutation_p(full, part) if full and part else float("nan"),
        "specificity": None, "specificity_p": float("nan"), "n_perturbed": 0,
    }

    # Perturbation results live in their own files; match them to this verifier by tag.
    suffix = b2_path.name.replace(f"b2_{dataset}_dev", "").replace(".jsonl", "")
    pert = CKPT / f"perturbation_{dataset}_dev{suffix}.jsonl"
    if pert.exists():
        prows = [json.loads(l) for l in open(pert) if l.strip()]
        spec = [(r["scores"]["control"] - r["scores"]["targeted"])
                - (r["scores"]["control"] - r["scores"]["random"])
                for r in prows
                if all(r["scores"].get(c) is not None
                       for c in ("control", "targeted", "random"))]
        if spec:
            out["specificity"] = round(st.mean(spec), 4)
            out["specificity_p"] = _paired_p(spec)
            out["n_perturbed"] = len(spec)
    return out


def main():
    ap = argparse.ArgumentParser(description="Master verifier-comparison table")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], default=None)
    ap.add_argument("--out", default=str(RESULTS / "verifier_comparison.json"))
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else ["finqa", "tatqa"]
    report, lines = {}, []
    for dataset in datasets:
        found = []
        for path in sorted(glob.glob(str(CKPT / f"b2_{dataset}_dev*.jsonl"))):
            path = Path(path)
            suffix = path.name.replace(f"b2_{dataset}_dev", "").replace(".jsonl", "")
            if suffix.startswith("verdicts") or "SPOT" in suffix:
                continue
            summary = RESULTS / f"b2_{dataset}_dev{suffix}.json"
            label = suffix.lstrip("_") or "nli-default"
            if summary.exists():
                cfg = json.loads(summary.read_text()).get("config", {})
                kind = cfg.get("verifier_kind", "nli")
                model = cfg.get("verifier")
                model = model[0] if isinstance(model, list) and model else model
                label = f"{kind}: {str(model).split('/')[-1][:26]}"
            found.append(analyse(dataset, path, label))
        report[dataset] = found

        lines.append(f"\n=== {dataset.upper()} ===")
        lines.append(f"{'verifier':34s} {'n':>4s} {'mean':>7s} {'separation':>12s} "
                     f"{'specificity':>13s} {'provenance':>12s}")
        lines.append("-" * 88)
        for row in found:
            lines.append(
                f"{row['verifier']:34s} {row['n']:4d} {row['mean']:7.3f} "
                f"{_fmt(row['separation'], row['separation_p']):>12s} "
                f"{_fmt(row['specificity'], row['specificity_p']):>13s} "
                f"{_fmt(row['provenance'], row['provenance_p']):>12s}")

    text = "\n".join(lines)
    print(text)
    print("\n  separation  = mean(correct) - mean(incorrect); does the score track answer quality")
    print("  specificity = targeted-removal drop - random-removal drop; does it track WHICH")
    print("                evidence was removed rather than how much")
    print("  provenance  = grounded - ungrounded vs deterministic operand provenance")
    print("  * p<0.05   ** p<0.01   *** p<0.001   (permutation tests, 20k resamples)")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[compare] wrote {args.out}")


if __name__ == "__main__":
    main()
