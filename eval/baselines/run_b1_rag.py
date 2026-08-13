"""
B1 baseline, end to end: retrieve -> generate -> score. This is the plain-RAG comparison floor
that B2 (adds RAGAS) and B3 (adds attribution + grounding + perturbation-validated
faithfulness) are measured against in Month 7.

It reuses `run_b1_retrieval.py`'s dataset loading, subsampling, index caching and checkpoint
handling, so both baselines score the identical fixed question set -- if the two scripts drew
different subsamples, the retrieval numbers from one couldn't explain the answer numbers from
the other.

    python eval/baselines/run_b1_rag.py --dataset finqa --split dev --limit 25
    python eval/baselines/run_b1_rag.py --dataset tatqa --split dev --generator stub  # offline
    python eval/baselines/run_b1_rag.py --dataset finqa --split dev --dry-run        # prompt only

**This costs nothing to run.** Generation defaults to a local open-weights model
(`generation.provider: local`) on your own GPU/MPS/CPU -- the project's zero-budget constraint
means no metered API sits in the default path. It still checkpoints after every batch, because
local generation is slow rather than expensive and a 250-question run is worth resuming rather
than restarting (CLAUDE.md guardrail).

Each per-question record carries both the retrieval outcome and the answer outcome, which is
what makes the failure taxonomy possible: gold chunk retrieved + wrong number is a generation
or arithmetic failure; gold chunk missed is a retrieval failure. CLAUDE.md asks for those to
stay separated in the logs, so they are separated at write time, not reconstructed later.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "src" / "ingestion", ROOT / "src" / "retrieval",
           ROOT / "src" / "generation", ROOT / "eval" / "metrics", ROOT / "eval" / "baselines"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from answer_metrics import aggregate_answers, score_answer  # noqa: E402
from config_utils import get as cfg_get, load_config, resolve_path  # noqa: E402
from embedder import Embedder, HashEmbedder  # noqa: E402
from generator import build_generator  # noqa: E402
from prompts import PROMPT_VERSION, build_messages  # noqa: E402
from retrieval_metrics import evaluate_one, gold_type_of, summarize  # noqa: E402
from retriever import Retriever  # noqa: E402
from run_b1_retrieval import (  # noqa: E402
    SEED,
    get_or_build_index,
    index_dir_for,
    load_checkpoint,
    load_dataset,
    subsample,
)

# Rough per-question cost at Claude Opus 5 list rates ($5/M input, $25/M output), assuming
# ~900 input tokens of evidence and ~500 output tokens including thinking. Order-of-magnitude
# only -- it exists so nobody starts a 250-question run without a number in front of them.
EST_COST_PER_QUESTION_USD = 0.017


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="B1 baseline: plain RAG, end to end")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--input", help="raw dataset json (default: from configs/config.yaml)")
    ap.add_argument("--limit", type=int, default=None,
                    help=f"questions to run (default: evaluation.subsample_size="
                         f"{cfg_get(cfg, 'evaluation.subsample_size', 250)})")
    ap.add_argument("--k", type=int, default=None, help="chunks passed to the generator")
    ap.add_argument("--scope", choices=["document", "corpus"], default=None)
    ap.add_argument("--generator", choices=["local", "api", "stub"], default=None,
                    help="default: generation.provider from config (local = free). "
                         "'api' bills a real account; 'stub' is an offline echo with "
                         "meaningless scores.")
    ap.add_argument("--embedder", choices=["model", "hash"], default="model")
    # Prompt/model ablation knobs. These vary what the *generator* does without touching the
    # config the frozen baseline reads, so an ablation arm can never silently become the
    # default -- `--tag` keeps each arm's results in its own file.
    ap.add_argument("--model", default=None,
                    help="override generation.local_model (ablation only)")
    ap.add_argument("--n-shot", type=int, default=None,
                    help="few-shot exemplars in the prompt (default: generation.n_shot)")
    ap.add_argument("--no-program", action="store_true",
                    help="direct answering: no answer_expression, no calculator (the v1 arm)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output files, e.g. --tag _pot3shot")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="4-bit NF4 weights (CUDA only). Needed to fit a 7B model on a free "
                         "Colab T4; ignored by the stub and API providers.")
    ap.add_argument("--no-4bit", action="store_true",
                    help="override generation.load_in_4bit=true for a laptop spot check "
                         "(bitsandbytes is CUDA-only). Pair with a smaller --model.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first prompt and exit without calling the API")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute answer scores and failure modes for an existing "
                         "checkpoint and rewrite the report. No generation, no GPU, no cost -- "
                         "use after changing a metric.")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    k = args.k or int(cfg_get(cfg, "retrieval.top_k", 5))
    scope = args.scope or cfg_get(cfg, "retrieval.scope", "document")
    limit = cfg_get(cfg, "evaluation.subsample_size", 250) if args.limit is None else args.limit

    raw_path = Path(args.input) if args.input else resolve_path(
        cfg_get(cfg, f"datasets.{args.dataset}.{args.split}")
    )
    if not raw_path.exists():
        sys.exit(f"[b1-rag] {raw_path} not found -- run: bash scripts/download_data.sh")

    print(f"[b1-rag] loading {args.dataset}/{args.split} from {raw_path}")
    chunks, questions = load_dataset(args.dataset, raw_path, args.split)
    questions = subsample(questions, limit or None)
    needed_docs = {q.doc_id for q in questions}
    if scope == "document":
        chunks = [c for c in chunks if c.doc_id in needed_docs]

    embedder = (HashEmbedder(max_tokens=cfg_get(cfg, "embedding.max_tokens_per_chunk", 256))
                if args.embedder == "hash" else Embedder.from_config(cfg))
    index_dir = index_dir_for(cfg, args.dataset, args.split, embedder, limit)
    index = get_or_build_index(chunks, embedder, index_dir, needed_docs, args.rebuild)
    retriever = Retriever(embedder=embedder, index=index, top_k=k, scope=scope)

    provider = args.generator or cfg_get(cfg, "generation.provider", "local")
    use_program = False if args.no_program else None    # None = take it from the config
    # None means "take it from the config"; the two flags are explicit overrides in either
    # direction, so a laptop spot check can turn 4-bit off without editing the frozen config.
    quantize = True if args.load_in_4bit else (False if args.no_4bit else None)
    generator = build_generator(cfg, provider=provider, model=args.model,
                                n_shot=args.n_shot, use_program=use_program,
                                load_in_4bit=quantize)
    version = getattr(generator, "prompt_version", PROMPT_VERSION)

    if args.dry_run:
        q = questions[0]
        hits = retriever.retrieve(q.question, doc_id=q.doc_id if scope == "document" else None)
        print(f"\n=== dry run: prompt for {q.qa_id} ({version}) ===")
        for msg in build_messages(q.question, hits,
                                  use_program=getattr(generator, "use_program", True),
                                  n_shot=getattr(generator, "n_shot", 3)):
            print(f"\n--- {msg['role']} ---\n{msg['content']}")
        print(f"\n[b1-rag] gold answer: {q.answer!r}  scale={q.scale!r}")
        print(f"[b1-rag] {len(questions)} questions would run; no API calls made")
        return

    suffix = "_STUBGEN" if provider == "stub" else ""
    if args.embedder == "hash":
        suffix += "_HASHEMB"
    suffix += args.tag
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"b1_rag_{args.dataset}_{args.split}{suffix}.jsonl"
    if args.fresh and ckpt_path.exists():
        ckpt_path.unlink()

    records, done = load_checkpoint(ckpt_path)
    todo = [q for q in questions if q.qa_id not in done]
    if done:
        print(f"[b1-rag] resuming: {len(done)} questions already done in {ckpt_path.name}")

    model_name = getattr(generator, "model", "?")
    if args.rescore:
        if not records:
            sys.exit(f"[b1-rag] --rescore needs an existing checkpoint; {ckpt_path} is empty")
        moved = rescore(records, questions)
        # The report must describe the run that produced the predictions, not the flags this
        # invocation happened to be called with -- otherwise a re-score silently relabels
        # whose numbers these are.
        first = records[0]["generated"]
        model_name = first.get("model", model_name)
        version = first.get("prompt_version", version)
        ckpt_path.write_text("".join(json.dumps(r) + "\n" for r in records))
        print(f"[b1-rag] re-scored {len(records)} records from {model_name} "
              f"({version}); {moved} correctness verdicts changed")
        todo = []

    if todo and provider == "local":
        print(f"[b1-rag] {len(todo)} questions through local model {generator.model} "
              f"(free). First run downloads the weights once. Ctrl-C is safe -- progress "
              f"checkpoints every {args.checkpoint_every} questions.")
    elif provider == "api":
        print(f"[b1-rag] {len(todo)} BILLED API calls to {generator.model}, rough estimate "
              f"~${len(todo) * EST_COST_PER_QUESTION_USD:.2f}. This project's zero-cost "
              f"default is --generator local.")

    chunk_types = {(c.doc_id, c.chunk_id): c.chunk_type for c in index.chunks}
    ks = sorted({1, min(3, k), k})
    t0 = time.time()
    n_errors = 0

    with open(ckpt_path, "a") as ckpt:
        for start in range(0, len(todo), args.checkpoint_every):
            batch = todo[start:start + args.checkpoint_every]
            hits_per_q = retriever.retrieve_batch(
                [q.question for q in batch],
                [q.doc_id if scope == "document" else None for q in batch],
                k=k,
            )
            for q, hits in zip(batch, hits_per_q):
                generated = generator.generate(q.question, hits)
                if generated.error:
                    n_errors += 1

                retrieved_ids = [h.chunk.chunk_id for h in hits]
                types = {cid: chunk_types.get((q.doc_id, cid)) for cid in q.gold_chunk_ids}
                retrieval_metrics = evaluate_one(retrieved_ids, q.gold_chunk_ids, ks)
                answer_scores = score_answer(
                    generated.answer, q.answer, answer_type=q.answer_type, scale=q.scale,
                    exe_answer=q.exe_answer, answer_is_percent=q.answer_is_percent,
                )
                # The failure taxonomy, decided at write time rather than reconstructed later.
                gold_retrieved = bool(set(q.gold_chunk_ids) & set(retrieved_ids))
                failure = _failure_mode(answer_scores, q.gold_chunk_ids, gold_retrieved,
                                        generated.insufficient_evidence, generated.error)

                record = {
                    "qa_id": q.qa_id,
                    "doc_id": q.doc_id,
                    "dataset": q.dataset,
                    "question": q.question,
                    "gold_answer": q.answer,
                    "scale": q.scale,
                    "answer_type": q.answer_type,
                    "gold_chunk_ids": q.gold_chunk_ids,
                    "gold_type": gold_type_of(q.gold_chunk_ids, types),
                    "retrieved": [h.to_json() for h in hits],
                    "metrics": retrieval_metrics,
                    "generated": generated.to_json(),
                    "answer_scores": answer_scores,
                    "gold_retrieved": gold_retrieved,
                    "failure_mode": failure,
                }
                records.append(record)
                ckpt.write(json.dumps(record) + "\n")
            ckpt.flush()
            print(f"[b1-rag] {min(start + len(batch), len(todo))}/{len(todo)} "
                  f"({time.time() - t0:.0f}s, {n_errors} errors)")

    report = summarize(records, ks)
    report["answers"] = aggregate_answers([r["answer_scores"] for r in records])
    report["failure_modes"] = _count(records, "failure_mode")
    report["declined_rate"] = round(
        sum(1 for r in records if r["generated"]["insufficient_evidence"]) / len(records), 4
    ) if records else 0.0
    report["citation_precision"] = _citation_precision(records)
    # Program-of-thought diagnostics. `program_rate` is the one that says whether the prompt
    # took effect at all -- a low accuracy with a low program rate is a prompting problem,
    # while a low accuracy with a high program rate is the model picking the wrong operands,
    # and those two call for completely different fixes.
    n = len(records) or 1
    report["program_rate"] = round(
        sum(1 for r in records if r["generated"].get("answer_source") == "program") / n, 4)
    report["unparseable_json_rate"] = round(
        sum(1 for r in records if r["generated"].get("error") == "unparseable_json") / n, 4)
    report["config"] = {
        "dataset": args.dataset,
        "split": args.split,
        "baseline": "B1_plain_rag",
        "scope": scope,
        "top_k": k,
        "embedding_model": embedder.model_name,
        "generation_model": model_name,
        "quantization": "nf4-4bit" if getattr(generator, "load_in_4bit", False) else "fp16",
        "prompt_version": version,
        "subsample_size": limit,
        "seed": SEED,
        "n_api_errors": n_errors,
    }
    if args.dataset == "tatqa":
        report["caveat"] = (
            "TAT-QA table-evidence gold_chunk_ids are heuristic (see "
            "tatqa_loader._table_gold_ids), so retrieval metrics and the 'retrieval' failure "
            "mode are approximate on table-sourced questions. Answer metrics are unaffected."
        )

    out_path = out_dir / f"b1_rag_{args.dataset}_{args.split}{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2))

    a = report["answers"]
    o = report["overall"]
    print(f"\n=== B1 end-to-end: {args.dataset}/{args.split} ===")
    print(f"  answers:   numeric_accuracy={a.get('numeric_accuracy')}  "
          f"span_em={a.get('span_exact_match')}  span_f1={a.get('span_token_f1')}  "
          f"(n={a.get('n_questions')}: {a.get('n_numeric')} numeric, {a.get('n_textual')} span)")
    print(f"  retrieval: recall@{k}={o.get(f'recall@{k}', 0):.3f}  "
          f"hit@{k}={o.get(f'hit@{k}', 0):.3f}")
    print(f"  failures:  {report['failure_modes']}")
    print(f"  declined:  {report['declined_rate']}   citation precision: "
          f"{report['citation_precision']}")
    print(f"  program:   answered via executed expression={report['program_rate']}  "
          f"unparseable_json={report['unparseable_json_rate']}  prompt={version}")
    print(f"\n[b1-rag] wrote {out_path}")
    print(f"[b1-rag] per-question records: {ckpt_path}")


def _failure_mode(answer_scores, gold_chunk_ids, gold_retrieved, declined, error):
    """One place that decides why a question failed, so the live loop and `--rescore` can
    never drift apart. Ordering matters: a missing gold chunk explains the failure even if
    the model also declined, and CLAUDE.md asks retrieval and generation failures to stay
    separated rather than being collapsed into one bucket."""
    if answer_scores["numeric_match"] or answer_scores["exact_match"]:
        return None
    if gold_chunk_ids and not gold_retrieved:
        return "retrieval"
    if declined:
        return "declined"
    if error:
        return "api_error"
    return "generation"


def rescore(records, questions) -> int:
    """Recompute answer scores and failure modes for already-generated records, in place.

    Generation is the expensive half and its outputs are already on disk, so a change to a
    *metric* must never require re-running a model. This is what makes correcting an
    evaluation bug free: the predictions are fixed, only the verdict on them changes.

    Returns the number of records whose correctness verdict actually moved, which is the
    number worth reporting when a metric changes.
    """
    by_id = {q.qa_id: q for q in questions}
    changed = 0
    for record in records:
        q = by_id.get(record["qa_id"])
        if q is None:
            continue
        before = bool(record["answer_scores"]["numeric_match"]
                      or record["answer_scores"]["exact_match"])
        scores = score_answer(
            record["generated"]["answer"], q.answer, answer_type=q.answer_type, scale=q.scale,
            exe_answer=q.exe_answer, answer_is_percent=q.answer_is_percent,
        )
        record["answer_scores"] = scores
        record["failure_mode"] = _failure_mode(
            scores, q.gold_chunk_ids, record["gold_retrieved"],
            record["generated"].get("insufficient_evidence"), record["generated"].get("error"),
        )
        changed += bool(scores["numeric_match"] or scores["exact_match"]) != before
    return changed


def _count(records, key) -> dict:
    counts = {}
    for r in records:
        counts[str(r.get(key))] = counts.get(str(r.get(key)), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _citation_precision(records) -> float | None:
    """Of the chunks the model cited, what share are gold evidence. A cheap first look at
    whether its self-reported citations mean anything -- the real answer needs the Month 5
    sentence-level grounding layer, and the gap between the two is itself a finding."""
    scored = [r for r in records if r["gold_chunk_ids"] and r["generated"]["cited_chunk_ids"]]
    if not scored:
        return None
    total = 0.0
    for r in scored:
        cited = r["generated"]["cited_chunk_ids"]
        hits = len(set(cited) & set(r["gold_chunk_ids"]))
        total += hits / len(cited)
    return round(total / len(scored), 4)


if __name__ == "__main__":
    main()
