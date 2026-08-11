# Explainable Financial RAG — Project Memory

This file is read automatically by Claude Code at the start of every session in this repo.
It exists so you don't have to re-explain the project each time you open a new terminal/session.

## Project

**Title:** Explainable Financial RAG — Retrieval Attribution & Faithfulness-Verified Explanations
for Financial Question Answering
**Type:** B.E. CS/AI-ML final-year major project, CBIT Hyderabad, 8-month timeline
**Status:** Month 1 (literature review) complete. Month 2 (data pipeline) complete — both
FinQA (`src/ingestion/finqa_loader.py`) and TAT-QA (`src/ingestion/tatqa_loader.py`) loaders
are done and tested (12/12 tests passing). Next up: Month 3, `src/retrieval/` (baseline B1).

## The two contributions (do not scope-creep beyond these)

1. **Retrieval Attribution** — quantify which query tokens/entities/numeric values drove a
   retrieved chunk's score. Perturb the query (occlusion / local-surrogate), re-score against the
   retriever's similarity function. Anchor methodology on RankingSHAP/Rank-LIME — NOT a naive
   off-the-shelf SHAP/LIME call. This is training-free / perturbation-based, never
   fine-tuning-based — do not let generated text (code comments, paper drafts, README, etc.)
   describe it as fine-tuning.
2. **Faithfulness-of-Faithfulness Testing** — run RAGAS, then perturb/remove the evidence it
   claims is important, and check whether the RAGAS score moves in the expected direction.
   Produces a labeled dataset of RAGAS failure cases.

A supporting **evidence-grounding** layer (sentence-level answer↔context matching) feeds both
contributions but is not itself a novelty claim.

## Base paper / closest related work

- **Base paper (architecture to extend):** Bayesian RAG — Ngartera, Nadarajah & Koina,
  *Frontiers in Artificial Intelligence*, 2025/2026. MC-Dropout uncertainty-penalized retrieval
  scoring on SEC 10-K filings. No source-attribution trail, no faithfulness-metric testing — this
  project extends it in both directions.
- **Closest peer-reviewed competitor to differentiate from:** FinRAG-12B (Katerenchuk, Duboue &
  Evanini, ACL 2026 Industry Track) — production banking RAG system with document-level citation
  tags and a static faithfulness sub-score. Proprietary fine-tune, not something to reproduce in
  code — cite it, don't build on it.

## Datasets (v1 scope — do not add SEC filings/other corpora without re-confirming scope)

- **FinQA** (Chen et al., 2021) — numerical reasoning over S&P 500 earnings reports; has gold
  supporting-fact indices (`gold_inds`) — use these directly for Precision@k/Recall@k, don't
  invent a new labeling scheme. Loader done: `src/ingestion/finqa_loader.py`.
- **TAT-QA** (Zhu et al., 2021) — hybrid tabular/textual financial QA. Loader done:
  `src/ingestion/tatqa_loader.py`, tested against the real dev set. Table-evidence
  `gold_chunk_ids` are a documented best-effort heuristic (no gold row index exists in the
  source data) — don't treat them as ground truth in eval code without re-reading
  `_table_gold_ids`'s docstring.

## Baselines

- B1 — Plain RAG (retriever + generator, no explanation layer)
- B2 — RAG + RAGAS only
- B3 — Full system (B2 + attribution + evidence grounding + perturbation-validated faithfulness)

## Tech stack

- Python 3.10+, pandas/NumPy, spaCy (sentence segmentation, financial NER)
- Embeddings: sentence-transformers (BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2)
- Retrieval: FAISS, single retriever only in v1 (no reranker, no BM25 hybrid — stretch goal only)
- Generation: one fixed instruction-tuned LLM via API, evidence-bound prompt
- Faithfulness: RAGAS (`ragas` package) + custom perturbation harness
- Demo: Streamlit (shell exists at `demo/streamlit_app.py`, cached resource loading already
  wired, not yet connected to a real pipeline); experiment tracking: MLflow or structured CSV/JSON

## Shared internal schema

Every dataset loader must output `Chunk` and `Question` from `src/ingestion/schema.py` — see
that file's docstring. Downstream code (retrieval, attribution, grounding) only ever imports
from `schema.py`, never from `finqa_loader.py`/`tatqa_loader.py` directly, so it stays
dataset-agnostic even though the two raw formats are quite different.

## Repo structure (Section 7 of brief)

```
finrag-explain/
  data/{raw,processed,sample}/   # raw+processed are gitignored; sample/ has tiny checked-in fixtures
  src/{ingestion,retrieval,attribution,grounding,generation,faithfulness}/
  eval/{baselines,metrics,results}/
  demo/streamlit_app.py
  notebooks/
  configs/config.yaml
  paper/
  scripts/download_data.sh
  tests/
  requirements.txt
  README.md
```

## Engineering guardrails — read before writing code in these areas

- **Attribution (Month 4):** never re-run the full RAG pipeline per perturbation. Pre-compute
  chunk embeddings once, batch-embed query perturbations, score via a single matrix multiply
  against cached embeddings. If you catch yourself calling the generator LLM inside a
  perturbation loop — stop, that's the compute-exhausting anti-pattern.
- **RAGAS (Month 6):** throttle concurrency (`RunConfig(max_workers=2, timeout=180,
  max_retries=5)`) to avoid 429 rate-limit death spirals. Run each perturbed case 3x, report
  mean/variance, not a single score — treat high variance as a finding, not noise.
- **FinQA table linearization:** map header→value explicitly per row (don't flatten tables into
  one string — this silently kills numeric retrieval accuracy). Already implemented this way in
  `finqa_loader.py::_linearize_table_row` — follow the same pattern for TAT-QA, adjusted for its
  multi-row headers. Enforce a hard token limit using the actual embedding model's tokenizer,
  not word/char counts.
- **TAT-QA multi-row headers:** don't assume `table[0]` is a single clean header row the way
  FinQA's is — inspect ~20 real examples first (see `tatqa_loader.py` docstring).
- **Numeric answer scoring:** normalize numeric strings (strip %, commas, currency; fix precision)
  before exact-match scoring — "5.2%" vs "5.2" vs "0.052" are the same answer. TAT-QA also has a
  `scale` field (thousand/million/percent) that changes what a bare number means — don't drop it.
- **Checkpoint everything.** Free-tier Colab/session disconnects mid-run are expected for the
  Month 6 evaluation loop — write results to disk after every N questions, resume from last
  checkpoint, never re-run a full loop from scratch after an interruption.
- **Pin dependency versions** in `requirements.txt` from Month 2 onward (ragas, transformers,
  faiss-cpu/gpu, embedding model version) — these libraries change APIs often enough to break
  reproducibility by Month 7 if left unpinned. `requirements.txt` currently has unpinned names
  as placeholders — pin them once the environment is confirmed working end-to-end.

## Working conventions

- Two separate loader modules for FinQA and TAT-QA (`finqa_loader.py`, `tatqa_loader.py`) — don't
  force one shared parser, the schemas differ enough that it's not worth it. They share only
  `schema.py`'s output dataclasses.
- Attribution/perturbation evaluation runs on a fixed subsample (200-300 questions per dataset,
  `configs/config.yaml: evaluation.subsample_size`), not the full test set — this is a
  scope/sanity cap, not (if architected per the guardrails above) a compute-survival requirement.
- Separate retrieval/grounding failures from generation/arithmetic failures when logging
  faithfulness results — a right-evidence-wrong-math case is not a grounding failure.
