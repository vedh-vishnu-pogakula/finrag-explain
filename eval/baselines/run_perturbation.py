"""
Contribution 2: does the faithfulness metric respond to the evidence the answer actually used?

For every question this runs three conditions -- unperturbed control, remove the load-bearing
sentence, remove an equal number of unrelated sentences -- and re-scores each with the cached
statements. The comparison that matters is `targeted` against `random`, not against `control`:
removing any evidence lowers a score, so only the *difference between what was removed* is
evidence about the metric.

    # free: local NLI verifier, all 250 questions, no GPU
    python eval/baselines/run_perturbation.py --dataset finqa --split dev

    # RAGAS's unmodified LLM verifier -- one call per question per condition, so Colab
    python eval/baselines/run_perturbation.py --dataset finqa --split dev \
        --verifier llm --limit 60 --out-tag _LLMVER

Statement decomposition is invariant under context perturbation and is read from the cache, so
the NLI sweep costs nothing. Re-running `ragas.evaluate()` per condition instead would be
thousands of LLM calls recomputing an identical result -- the anti-pattern CLAUDE.md forbids.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "src" / "faithfulness", ROOT / "src" / "generation",
           ROOT / "src" / "grounding", ROOT / "src" / "ingestion", ROOT / "src" / "retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config_utils import load_config  # noqa: E402
from grounder import Grounder  # noqa: E402
from perturb import CONTROL, RANDOM, TARGETED, build_conditions  # noqa: E402
from segmenter import evidence_sentences  # noqa: E402
from staged import StagedFaithfulness, cache_is_stale, load_statements  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Contribution 2: faithfulness perturbation audit")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--verifier", choices=["nli", "llm"], default="nli")
    ap.add_argument("--verifier-model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="seeds the random-removal arm")
    ap.add_argument("--model", default=None, help="judge model, --verifier llm only")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    ckpt = Path(args.out_dir) / "checkpoints"
    records = [json.loads(l) for l in
               open(ckpt / f"b1_rag_{args.dataset}_{args.split}.jsonl") if l.strip()]
    cache = load_statements(ckpt / f"statements_{args.dataset}_{args.split}.jsonl")
    if not cache:
        sys.exit("[perturb] no statement cache -- run run_b2.py --phase decompose first")
    if args.limit:
        records = records[:args.limit]

    judge = None
    if args.verifier == "llm":
        from generator import build_generator
        from ragas_local import LocalRagasLLM

        judge = LocalRagasLLM(build_generator(cfg, provider="local", model=args.model,
                                              load_in_4bit=False if args.no_4bit else None))
    overrides = {"verifier_model": args.verifier_model} if args.verifier_model else {}
    scorer = StagedFaithfulness.from_config(cfg, llm=judge, **overrides)
    grounder = Grounder.from_config(cfg)

    print(f"[perturb] {len(records)} questions | verifier={args.verifier} "
          f"({scorer.verifier_model if args.verifier == 'nli' else judge.name})")

    t0, rows, skipped = time.time(), [], 0
    for i, record in enumerate(records, start=1):
        entry = cache.get(record["qa_id"])
        if entry is None or not entry.statements or cache_is_stale(
                entry, record["generated"]["answer"]):
            skipped += 1
            continue
        grounding = grounder.ground(record["generated"]["answer"], record["retrieved"],
                                    record["question"],
                                    record["generated"].get("answer_expression"))
        sentences = evidence_sentences(record["retrieved"])
        conditions = build_conditions(sentences, grounding.support, seed=args.seed)
        if not conditions:
            skipped += 1
            continue

        scores = {}
        for condition in conditions:
            score = (scorer.verify_with_llm(entry.statements, condition.contexts)
                     if args.verifier == "llm"
                     else scorer.verify(entry.statements, condition.contexts))
            scores[condition.condition] = None if score.score != score.score else score.score
        rows.append({
            "qa_id": record["qa_id"], "correct": not record.get("failure_mode"),
            "n_statements": len(entry.statements),
            "n_removed": conditions[1].n_removed, "n_sentences": conditions[1].n_sentences,
            "scores": scores,
            "removed_targeted": conditions[1].to_json()["removed"],
        })
        if args.verifier == "llm" and i % 10 == 0:
            print(f"[perturb] {i}/{len(records)} ({time.time() - t0:.0f}s)")

    report = _summarize(rows)
    report["config"] = {
        "dataset": args.dataset, "split": args.split, "verifier": args.verifier,
        "verifier_model": scorer.verifier_model if args.verifier == "nli" else judge.name,
        "seed": args.seed, "n_questions": len(rows), "n_skipped": skipped,
        "seconds": round(time.time() - t0, 1),
    }

    out = Path(args.out_dir) / f"perturbation_{args.dataset}_{args.split}{args.out_tag}.json"
    out.write_text(json.dumps(report, indent=2))
    (ckpt / f"perturbation_{args.dataset}_{args.split}{args.out_tag}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    e = report["effects"]
    print(f"\n=== Perturbation audit: {args.dataset}/{args.split} "
          f"[{args.verifier}] (n={len(rows)}, {skipped} skipped) ===")
    print(f"  control                     {e['control']}")
    print(f"  remove load-bearing evidence {e['targeted']}   (drop {e['drop_targeted']:+.4f})")
    print(f"  remove random evidence       {e['random']}   (drop {e['drop_random']:+.4f})")
    print(f"\n  SPECIFICITY (targeted drop - random drop): {e['specificity']:+.4f}  "
          f"p={e['p_specificity']}")
    print("  Positive = the metric responds specifically to the evidence the answer used.")
    print(f"  Removed {e['mean_removed']:.2f} of {e['mean_sentences']:.1f} sentences per "
          f"question in both arms.")
    print(f"\n[perturb] wrote {out}")


def _paired_p(deltas, trials: int = 20000) -> float:
    """Sign-flip permutation test on the paired per-question difference.

    Paired, because every question is scored under all three conditions -- the between-question
    variance in faithfulness is large and would swamp the effect if the arms were compared as
    independent samples.
    """
    import random as _r

    if not deltas:
        return 1.0
    observed = abs(st.mean(deltas))
    rng = _r.Random(0)
    hits = sum(abs(st.mean([d if rng.random() < 0.5 else -d for d in deltas])) >= observed
               for _ in range(trials))
    return round(hits / trials, 5)


def _summarize(rows) -> dict:
    def mean(condition):
        vals = [r["scores"][condition] for r in rows if r["scores"].get(condition) is not None]
        return round(st.mean(vals), 4) if vals else None

    paired = [(r["scores"][CONTROL], r["scores"][TARGETED], r["scores"][RANDOM])
              for r in rows
              if all(r["scores"].get(c) is not None for c in (CONTROL, TARGETED, RANDOM))]
    drop_t = [c - t for c, t, _ in paired]
    drop_r = [c - x for c, _, x in paired]
    # Positive specificity = removing the load-bearing sentence hurts MORE than removing an
    # equal amount of unrelated evidence, which is what an evidence-sensitive metric must do.
    # Sign matters and is easy to get backwards: drop_* are already control-minus-condition,
    # so the targeted drop must be the larger of the two.
    spec = [t - r for t, r in zip(drop_t, drop_r)]

    return {
        "effects": {
            "control": mean(CONTROL), "targeted": mean(TARGETED), "random": mean(RANDOM),
            "drop_targeted": round(st.mean(drop_t), 4) if drop_t else 0.0,
            "drop_random": round(st.mean(drop_r), 4) if drop_r else 0.0,
            "specificity": round(st.mean(spec), 4) if spec else 0.0,
            "p_specificity": _paired_p(spec),
            "n_paired": len(paired),
            "mean_removed": round(st.mean([r["n_removed"] for r in rows]), 2) if rows else 0,
            "mean_sentences": round(st.mean([r["n_sentences"] for r in rows]), 2) if rows else 0,
        },
        "by_correctness": {
            label: {
                "n": len(group),
                "drop_targeted": round(st.mean([r["scores"][CONTROL] - r["scores"][TARGETED]
                                                for r in group]), 4) if group else None,
                "drop_random": round(st.mean([r["scores"][CONTROL] - r["scores"][RANDOM]
                                              for r in group]), 4) if group else None,
            }
            for label, group in [
                ("correct", [r for r in rows if r["correct"] and all(
                    r["scores"].get(c) is not None for c in (CONTROL, TARGETED, RANDOM))]),
                ("incorrect", [r for r in rows if not r["correct"] and all(
                    r["scores"].get(c) is not None for c in (CONTROL, TARGETED, RANDOM))]),
            ]
        },
        "per_question": rows,
    }


if __name__ == "__main__":
    main()
