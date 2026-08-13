"""
Evidence grounding over the saved B1 records: which retrieved sentence supports each answer.

**Runs on the laptop, needs no GPU, and costs nothing.** Generation is the expensive half and
its per-question outputs are already checkpointed, so grounding is a pure post-process over
`eval/results/checkpoints/b1_rag_*.jsonl` -- the same property that makes `--rescore` free.
On FinQA it does not load a model at all, because numeric answers are grounded by operand
provenance rather than by entailment (see src/grounding/segmenter.py for why).

    python eval/baselines/run_grounding.py --dataset finqa --split dev
    python eval/baselines/run_grounding.py --dataset tatqa --split dev --backend stub

What the report is for. Three numbers matter downstream, and they answer different questions:

  * `groundedness` / `supported_rate` -- how much of the answer the evidence actually
    supports. This is the Month 5 headline.
  * `grounded_vs_cited` -- agreement between the chunks the *model said* it used and the
    chunks its answer is actually supported by. A model can cite [1] while its answer is
    entailed only by [3]; that gap is a finding, not an error, and B3 is where it gets used.
  * `grounded_vs_gold` -- whether the supporting sentence sits in the annotated gold
    evidence. This is the closest thing to a correctness check the grounding layer has, and
    it is deliberately reported *per failure mode*: grounding that lands on gold while the
    answer is wrong is the right-evidence-wrong-arithmetic case CLAUDE.md asks to keep
    separate from a grounding failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "src" / "grounding", ROOT / "src" / "ingestion",
           ROOT / "src" / "retrieval", ROOT / "eval" / "metrics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config_utils import get as cfg_get, load_config  # noqa: E402
from grounder import Grounder  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Evidence grounding over saved B1 records")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--backend", choices=["nli", "stub"], default=None,
                    help="default: grounding.backend from config. 'stub' is offline and "
                         "lexical -- its scores are meaningless and the output is suffixed.")
    ap.add_argument("--limit", type=int, default=None, help="grounding is cheap; default all")
    ap.add_argument("--records", default=None, help="override the B1 checkpoint path")
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    backend = args.backend or cfg_get(cfg, "grounding.backend", "nli")
    records_path = Path(args.records) if args.records else (
        ROOT / "eval" / "results" / "checkpoints" / f"b1_rag_{args.dataset}_{args.split}.jsonl"
    )
    if not records_path.exists():
        sys.exit(f"[grounding] {records_path} not found -- run run_b1_rag.py first, or copy "
                 "the checkpoints back from Drive after a Colab run.")

    records = [json.loads(line) for line in open(records_path) if line.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"[grounding] {len(records)} records from {records_path.name}, backend={backend}")

    grounder = Grounder.from_config(cfg, backend=backend)
    items = [(r["generated"]["answer"], r["retrieved"], r["question"],
              r["generated"].get("answer_expression")) for r in records]

    t0 = time.time()
    results = grounder.ground_batch(items)
    elapsed = time.time() - t0
    print(f"[grounding] grounded in {elapsed:.1f}s "
          f"({elapsed / max(1, len(records)) * 1000:.0f} ms/question)")

    rows = []
    for record, result in zip(records, results):
        cited = set(record["generated"].get("cited_chunk_ids") or [])
        gold = set(record.get("gold_chunk_ids") or [])
        grounded = set(result.cited_chunk_ids)
        rows.append({
            "qa_id": record["qa_id"],
            "failure_mode": record.get("failure_mode"),
            "correct": not record.get("failure_mode"),
            "grounding": result.to_json(),
            "agreement": {
                # Jaccard rather than a hit rate: both sets are small and either can be
                # empty, and "did they pick the same evidence" is symmetric.
                "grounded_vs_cited": _jaccard(grounded, cited),
                "grounded_vs_gold": _jaccard(grounded, gold),
                "cited_vs_gold": _jaccard(cited, gold),
                "grounded_hits_gold": bool(grounded & gold),
            },
        })

    report = _summarize(rows)
    report["config"] = {
        "dataset": args.dataset, "split": args.split, "backend": backend,
        "scorer": results[0].scorer if results else None,
        "n_records": len(records), "seconds": round(elapsed, 1),
        "support_threshold": grounder.support_threshold,
        "source_records": str(records_path.relative_to(ROOT)),
    }

    suffix = "_STUBNLI" if backend == "stub" else ""
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    detail = out_dir / "checkpoints" / f"grounding_{args.dataset}_{args.split}{suffix}.jsonl"
    detail.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out_path = out_dir / f"grounding_{args.dataset}_{args.split}{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2))

    o = report["overall"]
    print(f"\n=== Grounding: {args.dataset}/{args.split} ===")
    print(f"  groundedness   {o['groundedness']:.3f}   supported_rate {o['supported_rate']:.3f}"
          f"   contradicted {o['contradicted_rate']:.3f}")
    print(f"  claims/question {o['mean_claims']:.2f}   evidence sentences/question "
          f"{o['mean_evidence_sentences']:.1f}   numeric claims {o['numeric_claim_share']:.0%}")
    print(f"  agreement: grounded~gold {o['grounded_vs_gold']:.3f}   "
          f"grounded~cited {o['grounded_vs_cited']:.3f}   cited~gold {o['cited_vs_gold']:.3f}")
    print(f"  grounded evidence hits gold on {o['grounded_hits_gold']:.1%} of questions")
    print("\n  by failure mode:")
    for mode, stats in report["by_failure_mode"].items():
        print(f"    {mode:12s} n={stats['n']:3d}  supported={stats['supported_rate']:.3f}  "
              f"hits_gold={stats['grounded_hits_gold']:.3f}")
    print(f"\n[grounding] wrote {out_path}")
    print(f"[grounding] per-question: {detail}")


def _jaccard(a: set, b: set) -> float | None:
    """None when both sides are empty -- there is nothing to agree or disagree about, and
    scoring it 1.0 would inflate the mean with questions that were never measured."""
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def _stats(rows) -> dict:
    grounding = [r["grounding"] for r in rows]
    claims = [c for g in grounding for c in g["support"]]
    return {
        "n": len(rows),
        "groundedness": round(_mean([g["groundedness"] for g in grounding]), 4),
        "supported_rate": round(_mean([g["supported_rate"] for g in grounding]), 4),
        "contradicted_rate": round(_mean([g["contradicted_rate"] for g in grounding]), 4),
        "mean_claims": round(_mean([g["n_claims"] for g in grounding]), 4),
        "mean_evidence_sentences": round(_mean([g["n_evidence_sentences"] for g in grounding]), 4),
        "numeric_claim_share": round(
            sum(c["kind"] == "numeric" for c in claims) / len(claims), 4) if claims else 0.0,
        "grounded_vs_gold": round(_mean([r["agreement"]["grounded_vs_gold"] for r in rows]), 4),
        "grounded_vs_cited": round(_mean([r["agreement"]["grounded_vs_cited"] for r in rows]), 4),
        "cited_vs_gold": round(_mean([r["agreement"]["cited_vs_gold"] for r in rows]), 4),
        "grounded_hits_gold": round(_mean([r["agreement"]["grounded_hits_gold"] for r in rows]), 4),
    }


def _summarize(rows) -> dict:
    modes = {}
    for row in rows:
        modes.setdefault(str(row["failure_mode"]), []).append(row)
    return {
        "overall": _stats(rows),
        # Grounding that lands on gold evidence while the answer is wrong is the
        # right-evidence-wrong-arithmetic case, and it is only visible split this way.
        "by_failure_mode": {mode: _stats(group)
                            for mode, group in sorted(modes.items(),
                                                      key=lambda kv: -len(kv[1]))},
    }


if __name__ == "__main__":
    main()
