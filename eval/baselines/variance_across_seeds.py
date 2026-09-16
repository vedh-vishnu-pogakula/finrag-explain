"""
How stable is the perturbation audit? Variance across the only source of randomness it has.

The brief asks for every perturbed case to be run three times with mean and variance
reported. Read literally, that is a no-op here, and saying so is more honest than reporting
a variance of zero as if it had been measured: generation is greedy (`do_sample=False`),
the NLI verifier is a deterministic forward pass, and the targeted arm removes a sentence
chosen by deterministic operand lookup. Re-running any of them reproduces the same bytes.

The one stochastic component is the **random-removal arm** -- which sentences get removed
in the control condition the targeted drop is compared against. So that is what is varied:
the audit is re-run under several seeds, and this script reports

  * per seed: mean specificity (targeted drop minus random drop) and its paired p-value,
  * across seeds: mean, standard deviation and range of that specificity,
  * a check that control and targeted scores are byte-identical across seeds (they must be;
    if they are not, something nondeterministic has crept into the pipeline).

If the specificity's sign and significance hold under every seed, the finding does not depend
on which random sentences happened to be drawn.

    for s in 1 2 3; do
      python eval/baselines/run_perturbation.py --dataset finqa --seed $s --out-tag _SEED$s
    done
    python eval/baselines/variance_across_seeds.py
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval" / "results"
CKPT = RESULTS / "checkpoints"


def _paired_p(deltas, trials: int = 20000) -> float:
    observed, rng, hits = abs(st.mean(deltas)), random.Random(0), 0
    for _ in range(trials):
        hits += abs(st.mean([d if rng.random() < 0.5 else -d for d in deltas])) >= observed
    return hits / trials


def _rows(path: Path) -> dict:
    return {r["qa_id"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


def _specificity(rows: dict) -> list:
    return [(r["scores"]["control"] - r["scores"]["targeted"])
            - (r["scores"]["control"] - r["scores"]["random"])
            for r in rows.values()
            if all(r["scores"].get(c) is not None for c in ("control", "targeted", "random"))]


def analyse(dataset: str) -> dict:
    base_path = CKPT / f"perturbation_{dataset}_dev.jsonl"
    if not base_path.exists():
        raise SystemExit(f"[variance] {base_path} missing -- run run_perturbation.py first")
    runs = {0: _rows(base_path)}
    for path in sorted(glob.glob(str(CKPT / f"perturbation_{dataset}_dev_SEED*.jsonl"))):
        seed = int(re.search(r"_SEED(\d+)\.jsonl$", path).group(1))
        runs[seed] = _rows(Path(path))
    if len(runs) < 2:
        raise SystemExit(f"[variance] only the seed-0 run exists for {dataset}; run the "
                         "--seed sweep in the module docstring first")

    # Determinism check: everything except the random arm must be identical across seeds.
    base = runs[0]
    fixed_arms_identical = all(
        r["scores"]["control"] == base[q]["scores"]["control"]
        and r["scores"]["targeted"] == base[q]["scores"]["targeted"]
        for seed, rows in runs.items() if seed != 0
        for q, r in rows.items() if q in base)
    random_changed = {
        seed: sum(1 for q, r in rows.items()
                  if q in base and r["scores"]["random"] != base[q]["scores"]["random"])
        for seed, rows in runs.items() if seed != 0}

    per_seed = {}
    for seed, rows in sorted(runs.items()):
        spec = _specificity(rows)
        per_seed[seed] = {"n": len(spec), "specificity": round(st.mean(spec), 4),
                          "p": _paired_p(spec),
                          "random_drop": round(st.mean(r["scores"]["control"] - r["scores"]["random"]
                                                       for r in rows.values()
                                                       if r["scores"].get("random") is not None), 4)}
    means = [v["specificity"] for v in per_seed.values()]
    return {
        "n_seeds": len(runs),
        "per_seed": per_seed,
        "specificity_mean": round(st.mean(means), 4),
        "specificity_sd": round(st.stdev(means), 4) if len(means) > 1 else 0.0,
        "specificity_min": min(means), "specificity_max": max(means),
        "all_seeds_significant_05": all(v["p"] < 0.05 for v in per_seed.values()),
        "control_and_targeted_identical_across_seeds": fixed_arms_identical,
        "questions_whose_random_arm_changed": random_changed,
    }


def main():
    ap = argparse.ArgumentParser(description="Perturbation-audit variance across seeds")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], default=None)
    ap.add_argument("--out", default=str(RESULTS / "perturbation_variance.json"))
    args = ap.parse_args()

    report = {}
    for dataset in ([args.dataset] if args.dataset else ["finqa", "tatqa"]):
        rep = analyse(dataset)
        report[dataset] = rep
        print(f"\n=== {dataset.upper()} — NLI verifier, {rep['n_seeds']} seeds ===")
        print(f"{'seed':>4s} {'n':>4s} {'specificity':>12s} {'p':>8s} {'random drop':>12s}")
        for seed, v in rep["per_seed"].items():
            print(f"{seed:4d} {v['n']:4d} {v['specificity']:+12.4f} {v['p']:8.4f} "
                  f"{v['random_drop']:+12.4f}")
        print(f"  specificity across seeds: {rep['specificity_mean']:+.4f} ± "
              f"{rep['specificity_sd']:.4f}  (range {rep['specificity_min']:+.4f} .. "
              f"{rep['specificity_max']:+.4f})")
        print(f"  significant at 0.05 under every seed: {rep['all_seeds_significant_05']}")
        print(f"  control/targeted byte-identical across seeds: "
              f"{rep['control_and_targeted_identical_across_seeds']}  "
              f"(random arm changed on {rep['questions_whose_random_arm_changed']} questions)")
        if not rep["control_and_targeted_identical_across_seeds"]:
            print("  ** WARNING: deterministic arms differ across seeds -- investigate before "
                  "reporting anything from this audit")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[variance] wrote {args.out}")


if __name__ == "__main__":
    main()
