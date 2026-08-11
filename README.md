# Explainable Financial RAG

Retrieval attribution and faithfulness-verified explanations for financial question answering.
B.E. CS/AI-ML final-year major project, CBIT Hyderabad.

See `CLAUDE.md` for the full project context (contributions, base paper, datasets, guardrails) —
Claude Code reads it automatically at the start of every session in this repo.

## Status

- [x] Month 1 — Literature review, scope locked
- [x] Month 2 — Data pipeline
  - [x] FinQA loader (`src/ingestion/finqa_loader.py`) — tested, produces `chunks.jsonl` +
        `questions.jsonl` with gold evidence IDs matched to FinQA's own `gold_inds`
  - [x] TAT-QA loader (`src/ingestion/tatqa_loader.py`) — tested against the real dev set;
        handles multi-row headers and section-label rows (see module docstring for the
        header/data-row split heuristic and its known limitations)
- [ ] Month 3 — Baseline RAG end-to-end (B1) (**next**)
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

## Run the loaders

```bash
cd src/ingestion
python finqa_loader.py --input ../../data/raw/finqa/dev.json --split dev --out-dir ../../data/processed
python tatqa_loader.py --input ../../data/raw/tatqa/dev.json --split dev --out-dir ../../data/processed
```

Writes `data/processed/{finqa,tatqa}_dev_chunks.jsonl` and `_questions.jsonl`. Each question's
`gold_chunk_ids` line up with `chunk_id`s from the same `doc_id`:
- FinQA: verified exactly against the dataset's own `gold_inds` (chunk text for `table_3` in doc
  `V/2008/page_17.pdf-1` matches FinQA's gold string word-for-word).
- TAT-QA: paragraph evidence (`para_N`) comes directly from `rel_paragraphs`; table evidence has
  no gold row index in the source data, so `gold_chunk_ids` for table-sourced answers is a
  best-effort heuristic (row's linearized text contains the answer string verbatim) — see
  `_table_gold_ids` in `tatqa_loader.py`, and don't mistake it for a ground-truth label later.

## Tests

```bash
pytest tests/ -v
```

12 tests, covering both loaders against checked-in sample fixtures (no network needed):
question/chunk counts, gold IDs resolving to real chunks, table rows staying header-mapped (not
flattened into one string), a regression test for a header-merge bug found during Session 1
(year-label rows were briefly misclassified as data rows), and chunk deduping.

## Next session

Both loaders now produce `chunks.jsonl` — that's the Month 2 milestone (brief Section 9) met.
Move on to `src/retrieval/`: embed chunks with the model in `configs/config.yaml`, build a FAISS
index, and get a plain top-k retriever working — that's the Month 3 baseline (B1).

Known rough edge worth revisiting before trusting TAT-QA numbers in eval: the header-merge
heuristic in `tatqa_loader.py::_merge_header_cells` can drop a wide title that spans multiple
year columns (e.g. "Years Ended September 30," sometimes only attaches to one of the three year
columns it should cover) — doesn't affect correctness of the values themselves, just how much
header context survives into the linearized text. Fine for v1; flag if attribution/retrieval
quality on TAT-QA looks off later and this is worth a second pass.
