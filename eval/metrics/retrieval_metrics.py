"""
Retrieval metrics for baseline B1 (and the comparison floor every later baseline is scored
against).

Relevance is binary and comes straight from `Question.gold_chunk_ids` -- FinQA's own
`gold_inds` and TAT-QA's `rel_paragraphs`. Per CLAUDE.md we do not invent a labelling scheme
on top of those.

Two reporting choices worth knowing about before quoting any number:

* **Recall@k is the headline, not Precision@k.** FinQA questions have 1-4 gold facts out of
  ~30-60 chunks, so P@5 is bounded above by roughly (n_gold / 5) and a perfect retriever
  still scores ~0.4. Precision is reported for completeness; recall (and hit-rate) are what
  actually say whether the generator was given what it needed.

* **TAT-QA table-evidence gold ids are heuristic** (`tatqa_loader._table_gold_ids`: a row
  counts as gold if the answer string appears in it verbatim). `by_gold_type` breaks results
  into text-only / table-only / mixed precisely so a TAT-QA table number can be read with the
  caveat attached instead of being averaged into one headline figure that hides it.
"""
from __future__ import annotations

from collections import defaultdict


def evaluate_one(retrieved_ids: list[str], gold_ids: list[str], ks: list[int]) -> dict:
    """Metrics for a single question. `retrieved_ids` is rank-ordered (best first)."""
    gold = set(gold_ids)
    if not gold:
        return {}
    out = {}
    for k in ks:
        topk = retrieved_ids[:k]
        n_hit = len(gold.intersection(topk))
        out[f"precision@{k}"] = n_hit / k if k else 0.0
        out[f"recall@{k}"] = n_hit / len(gold)
        out[f"hit@{k}"] = 1.0 if n_hit else 0.0
        # "did the generator get *everything* it needed" -- the strict version of recall,
        # and the one that predicts whether a downstream arithmetic answer can even be right
        out[f"full_recall@{k}"] = 1.0 if gold.issubset(topk) else 0.0
    out["mrr"] = next(
        (1.0 / (i + 1) for i, cid in enumerate(retrieved_ids) if cid in gold), 0.0
    )
    out["n_gold"] = len(gold)
    return out


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def aggregate(per_question: list[dict]) -> dict:
    """Macro-average over questions (each question counts once, regardless of how many gold
    facts it has)."""
    if not per_question:
        return {}
    keys = [k for k in per_question[0] if k != "n_gold"]
    agg = {k: _mean([q[k] for q in per_question if k in q]) for k in keys}
    agg["n_questions"] = len(per_question)
    agg["mean_n_gold"] = _mean([q.get("n_gold", 0) for q in per_question])
    return agg


def gold_type_of(gold_ids: list[str], chunk_types: dict[str, str]) -> str:
    """'text' | 'table' | 'mixed' | 'unknown' -- what kind of evidence this question needs.
    `chunk_types` maps chunk_id -> Chunk.chunk_type for that question's document."""
    types = {chunk_types.get(cid) for cid in gold_ids}
    types.discard(None)
    if not types:
        return "unknown"
    if types == {"text"}:
        return "text"
    if types == {"table_row"}:
        return "table"
    return "mixed"


def summarize(records: list[dict], ks: list[int]) -> dict:
    """Roll per-question records into the report written to eval/results/.

    Each record needs: qa_id, dataset, gold_type, metrics (from evaluate_one), and optionally
    answer_type. Questions with no gold evidence are counted and excluded -- averaging them in
    as zeros would understate the retriever, and silently dropping them would overstate how
    much of the split was actually evaluated.
    """
    scored = [r for r in records if r.get("metrics")]
    skipped = len(records) - len(scored)

    report = {
        "overall": aggregate([r["metrics"] for r in scored]),
        "n_questions_total": len(records),
        "n_questions_scored": len(scored),
        "n_questions_skipped_no_gold": skipped,
        "ks": ks,
    }

    for field in ("dataset", "gold_type", "answer_type"):
        buckets = defaultdict(list)
        for r in scored:
            value = r.get(field)
            if value is not None:
                buckets[value].append(r["metrics"])
        if buckets:
            report[f"by_{field}"] = {
                name: aggregate(ms) for name, ms in sorted(buckets.items())
            }
    return report


def format_report(report: dict, headline_k: int = 5) -> str:
    """Human-readable summary for stdout -- the JSON on disk stays the source of truth."""
    lines = []
    o = report.get("overall", {})
    lines.append(
        f"scored {report.get('n_questions_scored', 0)}/{report.get('n_questions_total', 0)} "
        f"questions ({report.get('n_questions_skipped_no_gold', 0)} skipped: no gold evidence)"
    )
    lines.append(
        f"  recall@{headline_k}={o.get(f'recall@{headline_k}', 0):.3f}  "
        f"full_recall@{headline_k}={o.get(f'full_recall@{headline_k}', 0):.3f}  "
        f"hit@{headline_k}={o.get(f'hit@{headline_k}', 0):.3f}  "
        f"precision@{headline_k}={o.get(f'precision@{headline_k}', 0):.3f}  "
        f"mrr={o.get('mrr', 0):.3f}"
    )
    for field, label in (("by_gold_type", "gold evidence type"), ("by_answer_type", "answer type")):
        if field in report:
            lines.append(f"  by {label}:")
            for name, m in report[field].items():
                lines.append(
                    f"    {name:<12} n={m['n_questions']:<5} "
                    f"recall@{headline_k}={m.get(f'recall@{headline_k}', 0):.3f}  "
                    f"mrr={m.get('mrr', 0):.3f}"
                )
    return "\n".join(lines)
