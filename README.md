# Explainable Financial RAG

Retrieval attribution and faithfulness-verified explanations for financial question answering.
B.E. CS/AI-ML final-year major project, CBIT Hyderabad.

See `CLAUDE.md` for the full project context (contributions, base paper, datasets, guardrails) —
Claude Code reads it automatically at the start of every session in this repo.

## Status

- [x] Month 1 — Literature review, scope locked
- [ ] Month 2 — Data pipeline (**in progress**)
  - [x] FinQA loader (`src/ingestion/finqa_loader.py`) — tested, produces `chunks.jsonl` +
        `questions.jsonl` with gold evidence IDs matched to FinQA's own `gold_inds`
  - [ ] TAT-QA loader (`src/ingestion/tatqa_loader.py`) — scaffolded, schema documented, not
        yet implemented — see its docstring for the exact plan
- [ ] Month 3 — Baseline RAG end-to-end (B1)
- [ ] Month 4 — Retrieval attribution (Contribution 1)
- [ ] Month 5 — Evidence grounding + RAGAS integration (B2)
- [ ] Month 6 — Faithfulness perturbation testing (Contribution 2)
- [ ] Month 7 — Full evaluation (B1/B2/B3) + Streamlit demo + human study
- [ ] Month 8 — Paper write-up, workshop submission

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Get the data

```bash
bash scripts/download_data.sh      # downloads FinQA + TAT-QA into data/raw/ (gitignored)
```

Small 8-example fixtures for fast, offline testing already live in `data/sample/` (checked
into git) — you don't need the full download to run the tests below.

## Run the FinQA loader

```bash
cd src/ingestion
python finqa_loader.py --input ../../data/raw/finqa/dev.json --split dev --out-dir ../../data/processed
```

Writes `data/processed/finqa_dev_chunks.jsonl` and `finqa_dev_questions.jsonl`. Each question's
`gold_chunk_ids` line up directly with `chunk_id`s from the same `doc_id` — verified against
FinQA's real dev set: chunk text for `table_3` in doc `V/2008/page_17.pdf-1` matches FinQA's own
`gold_inds` string exactly.

## Tests

```bash
pytest tests/ -v
```

6 tests currently, covering the FinQA loader against the checked-in sample fixture (no network
needed): question/chunk counts, gold IDs resolving to real chunks, table rows staying
header-mapped (not flattened into one string), and chunk deduping.

## Next session

1. Implement `src/ingestion/tatqa_loader.py` — the docstring has the verified schema and a
   step-by-step TODO list. Write `tests/test_tatqa_loader.py` mirroring
   `tests/test_finqa_loader.py` against `data/sample/tatqa_sample.json`.
2. Once both loaders produce `chunks.jsonl`, that's the Month 2 milestone (brief Section 9) —
   move on to `src/retrieval/` (embed chunks, build a FAISS index) for the Month 3 baseline (B1).
