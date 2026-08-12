"""
Contribution 1 evaluation: run retrieval attribution over the fixed subsample and report
whether the explanations are faithful.

    python eval/baselines/run_attribution.py --dataset finqa --split dev --limit 50
    python eval/baselines/run_attribution.py --dataset tatqa --split dev --mode ranking
    python eval/baselines/run_attribution.py --dataset finqa --split dev --embedder hash  # fast

Costs nothing to run and never touches the generator -- attribution only ever re-scores query
variants against chunk embeddings that were computed once at index build time.

**All three methods share one `PerturbationScorer` per question.** That is the single most
important line in this file: occlusion, Shapley and the surrogate request overlapping
coalitions, and a shared scorer means each distinct one is embedded exactly once for the whole
question rather than once per method. The `cache_hit_rate` in the output records how much that
saved, typically 60-75%.

What the report answers, in order of importance to the paper:

1. Do the explanations beat random unit selection? (`lift_over_random` -- if this is ~0 the
   method is not explaining anything, and that would be the finding.)
2. Do the three methods agree with each other? (`method_agreement_spearman` -- disagreement
   means "the explanation" is method-dependent, which has to be stated.)
3. Where does attribution mass sit -- numbers, entities, content words, or stopwords?
   (`mean_mass_by_kind` -- the domain claim of Contribution 1.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "src" / "ingestion", ROOT / "src" / "retrieval",
           ROOT / "src" / "attribution", ROOT / "eval" / "metrics",
           ROOT / "eval" / "baselines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from attributors import build_attributor  # noqa: E402
from attribution_metrics import aggregate_attribution, faithfulness, mass_by_kind, spearman  # noqa: E402
from config_utils import get as cfg_get, load_config, resolve_path  # noqa: E402
from embedder import Embedder, HashEmbedder  # noqa: E402
from perturbation import PerturbationScorer, ranking_value_fn, score_value_fn  # noqa: E402
from retriever import Retriever  # noqa: E402
from segmentation import segment_query  # noqa: E402
from run_b1_retrieval import (  # noqa: E402
    SEED,
    get_or_build_index,
    index_dir_for,
    load_checkpoint,
    load_dataset,
    subsample,
)

METHODS = ("occlusion", "shapley", "surrogate")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Contribution 1: retrieval attribution eval")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--input")
    ap.add_argument("--limit", type=int, default=None,
                    help="questions to explain (default: evaluation.subsample_size)")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--mode", choices=["score", "ranking"], default="score",
                    help="score = explain the top chunk's similarity; "
                         "ranking = explain the whole top-k ordering (RankingSHAP)")
    ap.add_argument("--explain-target", choices=["top", "gold"], default="top",
                    help="which chunk to explain in score mode: the top-ranked one, or the "
                         "gold-evidence chunk when it was retrieved")
    ap.add_argument("--permutations", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--embedder", choices=["model", "hash"], default="model")
    ap.add_argument("--no-ner", action="store_true", help="disable entity grouping")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        build_attributor(method)          # fail fast on a typo, before any embedding work

    k = int(cfg_get(cfg, "retrieval.top_k", 5))
    scope = cfg_get(cfg, "retrieval.scope", "document")
    limit = cfg_get(cfg, "evaluation.subsample_size", 250) if args.limit is None else args.limit
    n_permutations = args.permutations or int(cfg_get(cfg, "attribution.n_permutations", 64))
    n_samples = args.samples or int(cfg_get(cfg, "attribution.n_samples", 256))
    fractions = tuple(cfg_get(cfg, "attribution.eval_fractions", [0.2, 0.5]))
    rbo_depth = int(cfg_get(cfg, "attribution.rbo_depth", 10))

    raw_path = Path(args.input) if args.input else resolve_path(
        cfg_get(cfg, f"datasets.{args.dataset}.{args.split}"))
    if not raw_path.exists():
        sys.exit(f"[attr] {raw_path} not found -- run: bash scripts/download_data.sh")

    print(f"[attr] loading {args.dataset}/{args.split}")
    chunks, questions = load_dataset(args.dataset, raw_path, args.split)
    questions = subsample(questions, limit or None)
    needed_docs = {q.doc_id for q in questions}
    if scope == "document":
        chunks = [c for c in chunks if c.doc_id in needed_docs]

    embedder = (HashEmbedder(max_tokens=cfg_get(cfg, "embedding.max_tokens_per_chunk", 256))
                if args.embedder == "hash" else Embedder.from_config(cfg))
    index = get_or_build_index(chunks, embedder,
                               index_dir_for(cfg, args.dataset, args.split, embedder, limit),
                               needed_docs, args.rebuild)
    retriever = Retriever(embedder=embedder, index=index, top_k=k, scope=scope)

    attributors = {
        "occlusion": build_attributor("occlusion"),
        "shapley": build_attributor("shapley", n_permutations=n_permutations, seed=SEED,
                                    exact_max_units=int(cfg_get(
                                        cfg, "attribution.exact_max_units", 10))),
        "surrogate": build_attributor("surrogate", n_samples=n_samples, seed=SEED),
    }

    suffix = "_HASHEMB" if args.embedder == "hash" else ""
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"attribution_{args.dataset}_{args.split}_{args.mode}{suffix}.jsonl"
    if args.fresh and ckpt_path.exists():
        ckpt_path.unlink()

    records, done = load_checkpoint(ckpt_path)
    todo = [q for q in questions if q.qa_id not in done]
    if done:
        print(f"[attr] resuming: {len(done)} questions already explained")
    print(f"[attr] explaining {len(todo)} questions, methods={methods}, mode={args.mode}")

    t0, n_skipped = time.time(), 0
    with open(ckpt_path, "a") as ckpt:
        for n, question in enumerate(todo, start=1):
            record = _explain_one(retriever, question, attributors, methods, args, scope,
                                  fractions, rbo_depth)
            if record is None:
                n_skipped += 1
                continue
            records.append(record)
            ckpt.write(json.dumps(record) + "\n")
            if n % args.checkpoint_every == 0:
                ckpt.flush()
                print(f"[attr] {n}/{len(todo)} ({time.time() - t0:.0f}s)")

    report = aggregate_attribution(records)
    report["config"] = {
        "dataset": args.dataset, "split": args.split, "mode": args.mode,
        "explain_target": args.explain_target, "methods": methods,
        "embedding_model": embedder.model_name, "top_k": k, "scope": scope,
        "subsample_size": limit, "seed": SEED, "n_permutations": n_permutations,
        "n_samples": n_samples, "eval_fractions": list(fractions),
        "ner_entity_grouping": not args.no_ner, "n_skipped": n_skipped,
        "contribution": "C1_retrieval_attribution",
    }
    out_path = out_dir / f"attribution_{args.dataset}_{args.split}_{args.mode}{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2))

    _print_report(report, fractions)
    print(f"\n[attr] wrote {out_path}")
    print(f"[attr] per-question records: {ckpt_path}")


def _explain_one(retriever, question, attributors, methods, args, scope, fractions, rbo_depth):
    """One question, one shared scorer, every method. Returns None if there is nothing to
    explain (an empty query, or gold-target mode where the gold chunk wasn't retrieved)."""
    units = segment_query(question.question, use_ner=not args.no_ner)
    if not units:
        return None

    doc_id = question.doc_id if scope == "document" else None
    scorer = PerturbationScorer(retriever, question.question, units, doc_id)
    base = scorer.base_scores()
    ranked = list(np.argsort(-base))

    if args.mode == "ranking":
        value_fn = ranking_value_fn(base, depth=rbo_depth)
        target_id, target_rank = None, None
    else:
        col = ranked[0]
        if args.explain_target == "gold":
            gold = {g for g in question.gold_chunk_ids}
            gold_cols = [c for c in ranked[:retriever.top_k]
                         if scorer.candidates[c].chunk_id in gold] if scorer.candidates else []
            if not gold_cols:
                return None       # gold not retrieved: nothing meaningful to attribute to
            col = gold_cols[0]
        value_fn = score_value_fn(int(col))
        target_id = scorer.candidates[int(col)].chunk_id
        target_rank = ranked.index(col)

    weights_by_method, payloads = {}, {}
    for method in methods:
        result = attributors[method].attribute(scorer, value_fn)
        weights_by_method[method] = result.weights
        payloads[method] = {
            "weights": [round(float(w), 6) for w in result.weights],
            "n_units": len(units),
            "faithfulness": faithfulness(scorer, value_fn, result.weights,
                                         fractions=fractions, seed=SEED),
            "mass_by_kind": mass_by_kind(units, result.weights),
            "meta": result.meta,
        }

    agreement = {}
    for a, b in combinations(methods, 2):
        agreement[f"{a}~{b}"] = spearman(weights_by_method[a], weights_by_method[b])

    return {
        "qa_id": question.qa_id,
        "doc_id": question.doc_id,
        "dataset": question.dataset,
        "question": question.question,
        "units": [u.to_json() for u in units],
        "target_chunk_id": target_id,
        "target_rank": target_rank,
        "gold_chunk_ids": question.gold_chunk_ids,
        "base_score": float(base[ranked[0]]),
        "methods": payloads,
        "agreement": agreement,
        "scorer_stats": scorer.stats(),
    }


def _print_report(report, fractions):
    print(f"\n=== Contribution 1: retrieval attribution "
          f"({report['config']['dataset']}/{report['config']['mode']}) ===")
    print(f"explained {report.get('n_questions', 0)} questions "
          f"({report['config']['n_skipped']} skipped)\n")

    fraction = fractions[0]
    header = (f"{'method':<11}{'comp↑':>9}{'rand':>9}{'lift↑':>9}"
              f"{'suff↓':>9}{'units':>8}")
    print(f"  at top-{int(fraction * 100)}% of units")
    print("  " + header)
    print("  " + "-" * len(header))
    for method, entry in report.get("by_method", {}).items():
        print(f"  {method:<11}"
              f"{_fmt(entry.get(f'norm_comprehensiveness@{fraction}')):>9}"
              f"{_fmt(entry.get(f'norm_random_comprehensiveness@{fraction}')):>9}"
              f"{_fmt(entry.get(f'lift_over_random@{fraction}')):>9}"
              f"{_fmt(entry.get(f'norm_sufficiency@{fraction}')):>9}"
              f"{_fmt(entry.get('mean_units_per_query')):>8}")

    print("\n  attribution mass by unit kind:")
    for method, entry in report.get("by_method", {}).items():
        shares = entry.get("mean_mass_by_kind") or {}
        rendered = "  ".join(f"{k}={v:.3f}" for k, v in shares.items())
        print(f"    {method:<11} {rendered}")

    if report.get("method_agreement_spearman"):
        print("\n  method agreement (Spearman):")
        for pair, value in report["method_agreement_spearman"].items():
            print(f"    {pair:<24} {value:+.3f}")


def _fmt(value):
    return "-" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
