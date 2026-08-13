"""
B2 baseline: B1 plus RAGAS faithfulness.

B2 exists to answer "does adding a faithfulness metric tell you anything B1 didn't", and it is
the object Contribution 2 later puts under test. It runs in two phases on purpose, because
they have completely different costs:

    --phase decompose   answer + question --LLM--> statements     ~500 calls, GPU, one-time
    --phase verify      statements + context --HHEM--> score      local, free, repeatable

Decomposition never reads the retrieved context (that is RAGAS's design, not an assumption --
`Faithfulness._create_statements` takes `row["response"]` and `row["user_input"]`), so its
output is invariant under the context perturbations Month 6 applies. Computing it once and
caching it is the difference between ~500 LLM calls and several thousand.

    # on Colab's free T4, with the frozen 7B judge -- resumable, checkpoints every 10
    python eval/baselines/run_b2.py --dataset finqa --split dev --phase decompose

    # anywhere, no GPU, seconds
    python eval/baselines/run_b2.py --dataset finqa --split dev --phase verify

    # laptop spot check with a small judge (scores are not comparable to the real run)
    python eval/baselines/run_b2.py --dataset finqa --phase decompose --limit 5 \
        --no-4bit --model Qwen/Qwen2.5-1.5B-Instruct --tag _SPOT

**Nothing here can reach a paid API.** The judge is constructed explicitly and every LLM call
happens inside `no_paid_providers()`, which strips paid credentials and substitutes an invalid
key so an unanticipated fallback fails with a 401 rather than an invoice. See
`src/faithfulness/ragas_local.py` for why that guard is not paranoia -- RAGAS's own factory
signature is `llm_factory(model, provider="openai")`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "src" / "faithfulness", ROOT / "src" / "generation",
           ROOT / "src" / "grounding", ROOT / "src" / "ingestion",
           ROOT / "src" / "retrieval", ROOT / "eval" / "metrics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config_utils import get as cfg_get, load_config  # noqa: E402
from staged import (  # noqa: E402
    StagedFaithfulness,
    append_statements,
    cache_is_stale,
    load_statements,
)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="B2 baseline: B1 + RAGAS faithfulness")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--phase", choices=["decompose", "verify", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--records", default=None, help="override the B1 checkpoint path")
    ap.add_argument("--model", default=None, help="judge model (ablation / spot checks only)")
    ap.add_argument("--no-4bit", action="store_true",
                    help="bitsandbytes is CUDA-only; use this for a laptop spot check")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--fresh", action="store_true", help="ignore the statement cache")
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    records_path = Path(args.records) if args.records else (
        ROOT / "eval" / "results" / "checkpoints" / f"b1_rag_{args.dataset}_{args.split}.jsonl"
    )
    if not records_path.exists():
        sys.exit(f"[b2] {records_path} not found -- run run_b1_rag.py first, or copy the "
                 "checkpoints back from Drive after a Colab run.")
    records = [json.loads(line) for line in open(records_path) if line.strip()]
    if args.limit:
        records = records[:args.limit]

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_path = ckpt_dir / f"statements_{args.dataset}_{args.split}{args.tag}.jsonl"
    if args.fresh and cache_path.exists():
        cache_path.unlink()

    print(f"[b2] {len(records)} records from {records_path.name}")

    if args.phase in ("decompose", "all"):
        _decompose(cfg, args, records, cache_path)
    if args.phase in ("verify", "all"):
        _verify(cfg, args, records, cache_path, out_dir)


def _decompose(cfg, args, records, cache_path) -> None:
    """Phase 1. The expensive half: one judge call per question, checkpointed and resumable."""
    from generator import build_generator
    from ragas_local import LocalRagasLLM

    cache = load_statements(cache_path)
    todo = [r for r in records
            if r["qa_id"] not in cache or cache_is_stale(cache[r["qa_id"]],
                                                         r["generated"]["answer"])]
    stale = sum(1 for r in records
                if r["qa_id"] in cache and cache_is_stale(cache[r["qa_id"]],
                                                          r["generated"]["answer"]))
    if cache:
        print(f"[b2] statement cache: {len(cache)} entries, {stale} stale "
              f"(answer changed since they were made)")
    if not todo:
        print("[b2] decomposition already complete; nothing to do")
        return

    quantize = False if args.no_4bit else None
    generator = build_generator(cfg, provider="local", model=args.model,
                                load_in_4bit=quantize)
    judge = LocalRagasLLM(generator,
                          max_new_tokens=int(cfg_get(cfg, "faithfulness.judge_max_new_tokens",
                                                     1024)))
    scorer = StagedFaithfulness.from_config(cfg, llm=judge)

    print(f"[b2] decomposing {len(todo)} answers with local judge {judge.name} "
          f"(free). Ctrl-C is safe -- checkpoints every {args.checkpoint_every}.")
    t0, errors = time.time(), 0
    for start in range(0, len(todo), args.checkpoint_every):
        batch = todo[start:start + args.checkpoint_every]
        entries = []
        for record in batch:
            entry = scorer.decompose(record["qa_id"], record["question"],
                                     record["generated"]["answer"])
            errors += bool(entry.error)
            entries.append(entry)
        append_statements(cache_path, entries)
        done = min(start + len(batch), len(todo))
        print(f"[b2] {done}/{len(todo)} ({time.time() - t0:.0f}s, {errors} errors)")
    print(f"[b2] statement cache -> {cache_path}")


def _verify(cfg, args, records, cache_path, out_dir) -> None:
    """Phase 2. Local, free, and the function Month 6 will call in a loop."""
    cache = load_statements(cache_path)
    if not cache:
        sys.exit(f"[b2] no statements at {cache_path}. Run --phase decompose first "
                 "(that half needs the judge model; this half does not).")

    scorer = StagedFaithfulness.from_config(cfg, llm=None)
    print(f"[b2] verifying {len(records)} answers against their contexts "
          f"(local NLI verifier, no LLM, no cost)")

    t0, rows, missing = time.time(), [], 0
    for record in records:
        entry = cache.get(record["qa_id"])
        if entry is None or cache_is_stale(entry, record["generated"]["answer"]):
            missing += 1
            continue
        contexts = [hit["text"] for hit in record["retrieved"]]
        score = scorer.verify(entry.statements, contexts)
        rows.append({
            "qa_id": record["qa_id"],
            "failure_mode": record.get("failure_mode"),
            "correct": not record.get("failure_mode"),
            "declined": bool(record["generated"].get("insufficient_evidence")),
            "n_statements": score.n_statements,
            "faithfulness": None if score.score != score.score else round(score.score, 4),
            "verdicts": score.verdicts,
        })
    elapsed = time.time() - t0
    if missing:
        print(f"[b2] {missing} records had no usable cached statements (run --phase decompose)")

    report = _summarize(rows)
    report["config"] = {
        "dataset": args.dataset, "split": args.split, "baseline": "B2_rag_plus_ragas",
        "metric": "ragas faithfulness (FaithfulnesswithHHEM)",
        "verifier": scorer.verifier_model,
        "verify_threshold": scorer.verify_threshold,
        "judge": _judge_name(cache),
        "n_records": len(rows), "n_missing_statements": missing,
        "seconds": round(elapsed, 1),
        "statement_cache": str(cache_path.relative_to(ROOT)),
    }

    out_path = out_dir / f"b2_{args.dataset}_{args.split}{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2))
    detail = out_dir / "checkpoints" / f"b2_{args.dataset}_{args.split}{args.tag}.jsonl"
    detail.write_text("".join(json.dumps(r) + "\n" for r in rows))

    o = report["overall"]
    print(f"\n=== B2: {args.dataset}/{args.split} ({elapsed:.1f}s) ===")
    print(f"  faithfulness   {o['faithfulness']}   scored {o['n_scored']}/{len(rows)}"
          f"   ({o['n_unscored']} had no statements)")
    print(f"  statements/answer {o['mean_statements']}")
    print("\n  by failure mode:")
    for mode, stats in report["by_failure_mode"].items():
        print(f"    {mode:12s} n={stats['n']:3d}  faithfulness={stats['faithfulness']}"
              f"  statements={stats['mean_statements']}")
    print(f"\n[b2] wrote {out_path}")
    print(f"[b2] per-question: {detail}")


def _judge_name(cache) -> list:
    """Which model(s) produced the cached statements. A B2 number is only interpretable
    alongside its judge, and more than one name here means the cache mixes runs -- which is
    worth seeing in the report rather than silently averaging over."""
    return sorted({e.judge for e in cache.values() if e.judge})


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def _stats(rows) -> dict:
    scored = [r for r in rows if r["faithfulness"] is not None]
    return {
        "n": len(rows),
        "n_scored": len(scored),
        # An answer with no statements (a decline, or an empty answer) is not unfaithful --
        # it asserted nothing. Scoring it 0.0 would misreport abstention as hallucination,
        # and the 7B model declines on 10% of TAT-QA.
        "n_unscored": len(rows) - len(scored),
        "faithfulness": _mean([r["faithfulness"] for r in scored]),
        "mean_statements": _mean([r["n_statements"] for r in rows]),
    }


def _summarize(rows) -> dict:
    modes = {}
    for row in rows:
        modes.setdefault(str(row["failure_mode"]), []).append(row)
    return {
        "overall": _stats(rows),
        # The comparison B2 exists to make: does faithfulness separate right answers from
        # wrong ones? If a wrong answer scores as faithful as a right one, the metric is
        # measuring something other than correctness -- which is exactly the gap
        # Contribution 2 investigates.
        "by_failure_mode": {mode: _stats(group) for mode, group
                            in sorted(modes.items(), key=lambda kv: -len(kv[1]))},
    }


if __name__ == "__main__":
    main()
