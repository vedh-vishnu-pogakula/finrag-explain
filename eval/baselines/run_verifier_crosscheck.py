"""
Does the substituted NLI verifier agree with RAGAS's own LLM verifier?

**This exists because B2 deviates from RAGAS's default, and a deviation has to be defended
with evidence rather than asserted.** RAGAS's `FaithfulnesswithHHEM` -- the variant that
replaces the LLM verifier with a model -- cannot run on this project's stack: Vectara's HHEM
ships custom remote code predating `transformers` 5.x and dies with

    AttributeError: 'HHEMv2ForSequenceClassification' object has no attribute
                    'all_tied_weights_keys'

and downgrading transformers is not available, because 5.15.0 produced every B1 number. So
stage 2 runs a standard NLI cross-encoder instead. The swap itself is one RAGAS sanctions;
the open question is whether *this particular* verifier reaches the same verdicts as the LLM
one on this data. That is what this script measures.

It reports three things, because they can disagree and each answers a different objection:

  * **score agreement** -- Pearson and Spearman between the two per-question faithfulness
    scores, plus mean absolute difference. Answers "would B2's headline number have been
    materially different".
  * **verdict agreement** -- statement-level accuracy and Cohen's kappa. Answers "do they
    agree case by case, or only on average". Two verifiers can match in mean while disagreeing
    on every question, and only kappa exposes that.
  * **disagreement examples** -- the actual statements they split on, with the LLM's stated
    reason, so the failure mode is inspectable rather than a number.

Cost: one LLM call per question, on a subset. That is precisely the property that makes the
LLM verifier unusable inside Month 6's perturbation loop and the NLI verifier necessary --
but it is perfectly affordable for 50 questions, which is all a cross-check needs.

    # on Colab, after the statement cache exists
    python eval/baselines/run_verifier_crosscheck.py --dataset finqa --limit 50
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "src" / "faithfulness", ROOT / "src" / "generation",
           ROOT / "src" / "grounding", ROOT / "src" / "ingestion",
           ROOT / "src" / "retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config_utils import load_config  # noqa: E402
from staged import StagedFaithfulness, cache_is_stale, load_statements  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="NLI verifier vs RAGAS's LLM verifier")
    ap.add_argument("--dataset", choices=["finqa", "tatqa"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=50,
                    help="questions to cross-check (default 50; the LLM half costs one call "
                         "each, so this is the knob that decides the runtime)")
    ap.add_argument("--model", default=None, help="judge model override (spot checks)")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    ckpt_dir = Path(args.out_dir) / "checkpoints"
    records_path = ckpt_dir / f"b1_rag_{args.dataset}_{args.split}.jsonl"
    cache_path = ckpt_dir / f"statements_{args.dataset}_{args.split}{args.tag}.jsonl"
    for path in (records_path, cache_path):
        if not path.exists():
            sys.exit(f"[crosscheck] {path} not found -- run run_b1_rag.py and "
                     "run_b2.py --phase decompose first.")

    records = [json.loads(line) for line in open(records_path) if line.strip()]
    cache = load_statements(cache_path)

    # Only questions with usable statements can be cross-checked: with none, both verifiers
    # return NaN and the pair carries no information about their agreement.
    usable = [r for r in records
              if (entry := cache.get(r["qa_id"])) is not None
              and entry.statements
              and not cache_is_stale(entry, r["generated"]["answer"])]
    usable = usable[:args.limit]
    if not usable:
        sys.exit("[crosscheck] no questions have cached statements; nothing to compare")

    from generator import build_generator
    from ragas_local import LocalRagasLLM

    quantize = False if args.no_4bit else None
    judge = LocalRagasLLM(build_generator(cfg, provider="local", model=args.model,
                                          load_in_4bit=quantize))
    scorer = StagedFaithfulness.from_config(cfg, llm=judge)

    print(f"[crosscheck] {len(usable)} questions | NLI verifier "
          f"{scorer.verifier_model} vs RAGAS LLM verifier {judge.name}")

    t0, rows, errors = time.time(), [], 0
    for i, record in enumerate(usable, start=1):
        statements = cache[record["qa_id"]].statements
        contexts = [hit["text"] for hit in record["retrieved"]]
        nli = scorer.verify(statements, contexts)
        try:
            llm = scorer.verify_with_llm(statements, contexts)
        except Exception as exc:                  # one bad call must not lose the run
            errors += 1
            print(f"[crosscheck] {record['qa_id']}: {type(exc).__name__}: {exc}")
            continue
        rows.append({
            "qa_id": record["qa_id"],
            "n_statements": nli.n_statements,
            "nli_score": nli.score,
            "llm_score": llm.score,
            "nli_verdicts": nli.verdicts,
            "llm_verdicts": llm.verdicts,
        })
        if i % 10 == 0:
            print(f"[crosscheck] {i}/{len(usable)} ({time.time() - t0:.0f}s, {errors} errors)")

    report = _summarize(rows)
    report["config"] = {
        "dataset": args.dataset, "split": args.split,
        "nli_verifier": scorer.verifier_model,
        "verify_threshold": scorer.verify_threshold,
        "llm_verifier": judge.name,
        "n_compared": len(rows), "n_errors": errors,
        "seconds": round(time.time() - t0, 1),
    }

    out_path = Path(args.out_dir) / f"verifier_crosscheck_{args.dataset}_{args.split}{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2))

    a = report["agreement"]
    print(f"\n=== Verifier cross-check: {args.dataset}/{args.split} (n={len(rows)}) ===")
    print(f"  per-question score:  pearson r={a['pearson']}  spearman={a['spearman']}"
          f"  mean|diff|={a['mean_abs_diff']}")
    print(f"  mean faithfulness:   NLI {a['mean_nli']}   LLM {a['mean_llm']}")
    print(f"  per-statement:       agreement={a['verdict_agreement']}  "
          f"kappa={a['cohens_kappa']}  (n={a['n_statements']})")
    print(f"  NLI says supported when LLM does not: {a['nli_only_supported']}")
    print(f"  LLM says supported when NLI does not: {a['llm_only_supported']}")
    if report["disagreements"]:
        print("\n  example disagreements:")
        for d in report["disagreements"][:4]:
            print(f"    NLI={d['nli']} LLM={d['llm']}  {d['statement'][:88]!r}")
            if d.get("reason"):
                print(f"        LLM reason: {d['reason'][:96]}")
    print(f"\n[crosscheck] wrote {out_path}")


def _pearson(xs, ys) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    # Zero variance on either side: correlation is undefined, not 0. It happens when every
    # question scores 1.0, and reporting 0.0 there would read as "the verifiers disagree".
    return round(num / (dx * dy), 4) if dx and dy else None


def _rank(values) -> list:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):                       # average ranks within ties
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _summarize(rows) -> dict:
    nli = [r["nli_score"] for r in rows]
    llm = [r["llm_score"] for r in rows]

    matched = both = nli_only = llm_only = 0
    disagreements = []
    for row in rows:
        # Align on statement text: the LLM echoes each statement back, and RAGAS occasionally
        # returns them in a different order or drops one, so positional zipping would compare
        # unrelated pairs and manufacture disagreement.
        llm_by_text = {v["statement"]: v for v in row["llm_verdicts"]}
        for verdict in row["nli_verdicts"]:
            counterpart = llm_by_text.get(verdict["statement"])
            if counterpart is None:
                continue
            matched += 1
            n_sup, l_sup = bool(verdict["supported"]), bool(counterpart["supported"])
            both += n_sup == l_sup
            nli_only += n_sup and not l_sup
            llm_only += l_sup and not n_sup
            if n_sup != l_sup and len(disagreements) < 25:
                disagreements.append({
                    "qa_id": row["qa_id"], "statement": verdict["statement"],
                    "nli": n_sup, "llm": l_sup,
                    "entailment": verdict.get("entailment"),
                    "reason": counterpart.get("reason"),
                })

    agreement = both / matched if matched else None
    # Cohen's kappa: agreement above what two verifiers would reach by chance given their
    # marginals. Raw agreement flatters any pair on a skewed dataset -- if 90% of statements
    # are supported, two verifiers that always say "supported" agree 90% of the time and have
    # learned nothing.
    kappa = None
    if matched:
        p_nli = sum(bool(v["supported"]) for r in rows for v in r["nli_verdicts"]
                    if v["statement"] in {x["statement"] for x in r["llm_verdicts"]}) / matched
        p_llm = sum(bool(v["supported"]) for r in rows for v in r["llm_verdicts"]
                    if v["statement"] in {x["statement"] for x in r["nli_verdicts"]}) / matched
        chance = p_nli * p_llm + (1 - p_nli) * (1 - p_llm)
        kappa = round((agreement - chance) / (1 - chance), 4) if chance < 1 else None

    return {
        "agreement": {
            "pearson": _pearson(nli, llm),
            "spearman": _pearson(_rank(nli), _rank(llm)) if len(nli) > 1 else None,
            "mean_abs_diff": round(sum(abs(a - b) for a, b in zip(nli, llm)) / len(nli), 4)
            if nli else None,
            "mean_nli": round(sum(nli) / len(nli), 4) if nli else None,
            "mean_llm": round(sum(llm) / len(llm), 4) if llm else None,
            "n_statements": matched,
            "verdict_agreement": round(agreement, 4) if agreement is not None else None,
            "cohens_kappa": kappa,
            "nli_only_supported": nli_only,
            "llm_only_supported": llm_only,
        },
        "disagreements": disagreements,
        "per_question": rows,
    }


if __name__ == "__main__":
    main()
