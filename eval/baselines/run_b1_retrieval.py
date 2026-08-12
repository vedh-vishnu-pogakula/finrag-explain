"""
B1 baseline -- retrieval half. Measures how often the plain dense retriever puts the gold
supporting facts in the top-k, per dataset, with no explanation layer of any kind.

This is the floor B2/B3 get compared against in Month 7, so it stays deliberately plain:
one dense retriever, no reranking, no query rewriting, no hybrid.

Usage:
    python eval/baselines/run_b1_retrieval.py --dataset finqa --split dev
    python eval/baselines/run_b1_retrieval.py --dataset tatqa --split dev --limit 50
    python eval/baselines/run_b1_retrieval.py --dataset finqa --split dev --embedder hash   # offline smoke

Design notes:

* **Subsample, then embed.** The fixed subsample (configs/config.yaml:
  `evaluation.subsample_size`) is drawn *first*, and only the documents those questions
  belong to get embedded. Embedding all 883 FinQA dev docs to evaluate 250 questions would
  be several minutes of pointless compute on every run.
* **Deterministic sample.** Seeded, and sorted by qa_id before sampling, so the same 250
  questions come back on every machine and every re-run -- B1 vs B3 comparisons in Month 7
  must be on identical question sets or the deltas mean nothing.
* **Checkpoint + resume** (CLAUDE.md guardrail). Per-question results append to a JSONL
  checkpoint every batch; re-running skips qa_ids already in it. Cheap here, but this is the
  harness Month 6's RAGAS loop reuses, where a mid-run disconnect is expected.
* **Cached index.** Chunk embeddings persist under `data/processed/index/`; a rerun reloads
  them. The index is rebuilt automatically if it was built with a different embedding model
  or is missing documents this run needs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "src" / "ingestion", ROOT / "src" / "retrieval",
           ROOT / "eval" / "metrics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from chunk_index import ChunkIndex  # noqa: E402
from config_utils import get as cfg_get, load_config, resolve_path  # noqa: E402
from embedder import Embedder, HashEmbedder  # noqa: E402
from retrieval_metrics import evaluate_one, gold_type_of, format_report, summarize  # noqa: E402
from retriever import Retriever  # noqa: E402

SEED = 13


def load_dataset(dataset: str, path: Path, split: str):
    """Run the appropriate loader over the raw file. Returns (chunks, questions) where chunks
    are deduped by (doc_id, chunk_id) -- both loaders re-emit a document's chunks once per
    question in that document."""
    if dataset == "finqa":
        from finqa_loader import build_chunk_index, load_finqa

        pairs = list(load_finqa(str(path), split))
    elif dataset == "tatqa":
        from tatqa_loader import build_chunk_index, load_tatqa

        pairs = list(load_tatqa(str(path)))
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    all_chunks, questions = [], []
    for chunks, question in pairs:
        all_chunks.extend(chunks)
        questions.append(question)
    return build_chunk_index(all_chunks), questions


def subsample(questions, n: int | None):
    """Deterministic fixed subsample -- see module docstring on why this must not drift.

    Shuffle once, then slice: this makes the subsamples *nested*, so the 25 questions from a
    quick `--limit 25` run are the first 25 of the full 250-question set. `random.sample(k=25)`
    and `random.sample(k=250)` on the same seed return unrelated sets, which would mean a cheap
    end-to-end spot check and the full retrieval eval silently scored different questions.
    """
    ordered = sorted(questions, key=lambda q: q.qa_id)
    random.Random(SEED).shuffle(ordered)
    return ordered if not n or n >= len(ordered) else ordered[:n]


def index_dir_for(cfg, dataset: str, split: str, embedder, limit) -> Path:
    """Index cache location. The subsample size is part of the path because a document-scoped
    index only contains the subsample's documents -- without it, a `--limit 25` run would
    overwrite the 250-question index and every alternating run would re-embed from scratch."""
    base = resolve_path(cfg_get(cfg, "retrieval.index_dir", "data/processed/index"))
    model_slug = embedder.model_name.replace("/", "_")
    size = limit or "all"
    return base / f"{dataset}_{split}_{model_slug}_n{size}"


def load_checkpoint(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # a run killed mid-write leaves one torn final line; drop it and continue
                # rather than refusing to resume
                break
    return records, {r["qa_id"] for r in records}


def get_or_build_index(chunks, embedder, index_dir: Path, needed_docs: set[str],
                       rebuild: bool) -> ChunkIndex:
    if not rebuild and (index_dir / "index_meta.json").exists():
        try:
            index = ChunkIndex.load(index_dir, expect_model=embedder.model_name)
            missing = needed_docs - set(index.doc_ids)
            if not missing:
                print(f"[b1] reusing cached index at {index_dir} ({len(index)} chunks)")
                return index
            print(f"[b1] cached index is missing {len(missing)} needed docs -- rebuilding")
        except (ValueError, OSError) as e:
            print(f"[b1] cached index unusable ({e}) -- rebuilding")

    print(f"[b1] embedding {len(chunks)} chunks with {embedder.model_name} ...")
    t0 = time.time()
    index = ChunkIndex.build(chunks, embedder, show_progress=True)
    index.save(index_dir)
    report = index.meta.get("truncation", {})
    print(f"[b1] built index in {time.time() - t0:.1f}s -> {index_dir}")
    print(f"[b1] token budget: cap={report.get('max_tokens_cap')} "
          f"p50={report.get('p50_tokens')} p95={report.get('p95_tokens')} "
          f"max={report.get('max_tokens_seen')} "
          f"truncated={report.get('n_truncated')} ({report.get('pct_truncated')}%)")
    return index


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="B1 baseline: plain dense retrieval eval")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--input", help="raw dataset json (default: from configs/config.yaml)")
    ap.add_argument("--limit", type=int, default=None,
                    help=f"questions to score (default: evaluation.subsample_size="
                         f"{cfg_get(cfg, 'evaluation.subsample_size', 250)}; 0 = all)")
    ap.add_argument("--ks", default="1,3,5,10", help="cutoffs to report")
    ap.add_argument("--scope", choices=["document", "corpus"], default=None)
    ap.add_argument("--embedder", choices=["model", "hash"], default="model",
                    help="'hash' = offline deterministic stub, for smoke tests only")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--rebuild", action="store_true", help="force re-embedding of chunks")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    top_k = max(ks)
    scope = args.scope or cfg_get(cfg, "retrieval.scope", "document")
    limit = cfg_get(cfg, "evaluation.subsample_size", 250) if args.limit is None else args.limit

    raw_path = Path(args.input) if args.input else resolve_path(
        cfg_get(cfg, f"datasets.{args.dataset}.{args.split}")
    )
    if not raw_path.exists():
        sys.exit(f"[b1] {raw_path} not found -- run: bash scripts/download_data.sh")

    print(f"[b1] loading {args.dataset}/{args.split} from {raw_path}")
    chunks, questions = load_dataset(args.dataset, raw_path, args.split)
    print(f"[b1] {len(questions)} questions, {len(chunks)} unique chunks")

    questions = subsample(questions, limit or None)
    needed_docs = {q.doc_id for q in questions}
    if scope == "document":
        # only embed what this subsample can actually retrieve from
        chunks = [c for c in chunks if c.doc_id in needed_docs]
    print(f"[b1] scoring {len(questions)} questions over {len(chunks)} chunks "
          f"({len(needed_docs)} docs), scope={scope}, k={ks}")

    if args.embedder == "hash":
        embedder = HashEmbedder(max_tokens=cfg_get(cfg, "embedding.max_tokens_per_chunk", 256))
        print("[b1] WARNING: offline hash-stub embedder -- results are a smoke test, "
              "not retrieval quality. Output filename is suffixed accordingly.")
    else:
        embedder = Embedder.from_config(cfg)

    index_dir = (Path(args.index_dir) if args.index_dir
                 else index_dir_for(cfg, args.dataset, args.split, embedder, limit))
    index = get_or_build_index(chunks, embedder, index_dir, needed_docs, args.rebuild)
    retriever = Retriever(embedder=embedder, index=index,
                          top_k=top_k, scope=scope)

    chunk_types = {(c.doc_id, c.chunk_id): c.chunk_type for c in index.chunks}

    suffix = "_HASHSTUB" if args.embedder == "hash" else ""
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"b1_retrieval_{args.dataset}_{args.split}{suffix}.jsonl"
    if args.fresh and ckpt_path.exists():
        ckpt_path.unlink()

    records, done = load_checkpoint(ckpt_path)
    if done:
        print(f"[b1] resuming: {len(done)} questions already scored in {ckpt_path.name}")
    todo = [q for q in questions if q.qa_id not in done]

    t0 = time.time()
    with open(ckpt_path, "a") as ckpt:
        for start in range(0, len(todo), args.checkpoint_every):
            batch = todo[start:start + args.checkpoint_every]
            hits_per_q = retriever.retrieve_batch(
                [q.question for q in batch],
                [q.doc_id if scope == "document" else None for q in batch],
                k=top_k,
            )
            for q, hits in zip(batch, hits_per_q):
                retrieved_ids = [h.chunk.chunk_id for h in hits]
                types = {cid: chunk_types.get((q.doc_id, cid)) for cid in q.gold_chunk_ids}
                record = {
                    "qa_id": q.qa_id,
                    "doc_id": q.doc_id,
                    "dataset": q.dataset,
                    "answer_type": q.answer_type,
                    "gold_chunk_ids": q.gold_chunk_ids,
                    "gold_type": gold_type_of(q.gold_chunk_ids, types),
                    "retrieved": [h.to_json() for h in hits],
                    "metrics": evaluate_one(retrieved_ids, q.gold_chunk_ids, ks),
                }
                records.append(record)
                ckpt.write(json.dumps(record) + "\n")
            ckpt.flush()
            print(f"[b1] {min(start + len(batch), len(todo))}/{len(todo)} "
                  f"({time.time() - t0:.1f}s)")

    report = summarize(records, ks)
    report["config"] = {
        "dataset": args.dataset,
        "split": args.split,
        "scope": scope,
        "embedding_model": embedder.model_name,
        "max_tokens_per_chunk": getattr(embedder, "max_tokens", None),
        "subsample_size": limit,
        "seed": SEED,
        "n_chunks_indexed": len(index),
        "n_docs": len(needed_docs),
        "index_truncation": index.meta.get("truncation"),
        "baseline": "B1_plain_retrieval",
    }
    if report.get("by_gold_type", {}).get("unknown"):
        report["gold_type_unknown_note"] = (
            "gold_type 'unknown' = none of the question's gold_chunk_ids resolve to a chunk in "
            "the index. On FinQA this is the documented header-row case (gold_inds citing "
            "'table_0'), ~0.5% of dev -- see finqa_loader.py's docstring. These questions "
            "cannot be answered by any retriever and are a fixed floor, not a regression."
        )
    if args.dataset == "tatqa":
        report["caveat"] = (
            "TAT-QA table-evidence gold_chunk_ids are a heuristic (answer-string match in the "
            "linearized row), not ground truth -- see tatqa_loader._table_gold_ids. Read "
            "by_gold_type['table'] and ['mixed'] with that in mind."
        )

    out_path = out_dir / f"b1_retrieval_{args.dataset}_{args.split}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print()
    print(f"=== B1 retrieval: {args.dataset}/{args.split} ({embedder.model_name}) ===")
    print(format_report(report, headline_k=5 if 5 in ks else ks[-1]))
    print(f"\n[b1] wrote {out_path}")
    print(f"[b1] per-question records: {ckpt_path}")


if __name__ == "__main__":
    main()
