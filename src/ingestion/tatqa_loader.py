"""
TAT-QA loader -- STUB. This is your Session 1 (or Session 2) task; finqa_loader.py in this
same directory is the fully-worked pattern to follow. Do NOT try to reuse finqa_loader.py's
parsing logic directly -- the schemas are different enough that CLAUDE.md calls this out
explicitly: two thin parsers, one shared output schema (schema.py).

Source: Zhu et al. 2021 (arxiv.org/abs/2105.07624), official data at
https://github.com/NExTplusplus/TAT-QA/tree/master/dataset_raw
(tatqa_dataset_train.json / _dev.json / _test.json).

Native shape (one dict per report excerpt -- confirmed by fetching tatqa_dataset_dev.json):
    {
      "table": {
          "uid": "...",
          "table": [[row0_col0, row0_col1, ...], [row1_col0, ...], ...]  # NOTE: often has
              # a multi-row header (e.g. row 0 = merged/blank cells, row 1 = year labels) --
              # see CLAUDE.md guardrail on this. Inspect ~20 examples before assuming a single
              # clean header row.
      },
      "paragraphs": [{"uid": "...", "order": 1, "text": "..."}, ...],
      "questions": [
          {
            "uid": "...", "question": "...", "answer": [...], "answer_type": "span"|"arithmetic"|...,
            "answer_from": "table"|"text"|"table-text",
            "rel_paragraphs": ["2"],       # 1-indexed into `paragraphs` (by "order"), when relevant
            "scale": "" | "thousand" | "million" | "percent" | ...
          },
          ...
      ]
    }

Key differences from FinQA that are why this needs its own parser (not a shared one):
  - No direct gold *row* index for table evidence -- questions point at `rel_paragraphs` for
    text evidence, but table evidence is implicit (answer_from == "table"/"table-text").
    You'll need to decide how to build gold_chunk_ids for table-sourced answers -- e.g. by
    string-matching the answer against linearized row text -- and document that choice,
    since (per CLAUDE.md) there's no ground truth for "correct attribution" here either.
  - Table headers can span multiple rows (see the "Years Ended September 30," / "2019" /
    "2018" / "2017" two-row header in a real example) -- don't assume table[0] is a clean
    single header row the way FinQA's table[0] is. Concatenate multi-row headers (e.g.
    "2019", building on the row above) before linearizing, per the CLAUDE.md guardrail.
  - `scale` matters for numeric answers ("1,496.5" with scale="million" is a different number
    than a bare "1,496.5") -- don't drop it when you get to numeric answer scoring.

TODO (Contribution-safe scope -- Month 2 milestone is just chunks.jsonl for both datasets):
  1. Write _merge_multirow_header(table) -> list[str] that concatenates header rows into one
     header label per column (e.g. "2019", "2018" from the example above).
  2. Write _linearize_table_row(...) analogous to finqa_loader's, reusing the same
     "the <row_label> of <col> is <val> ;" phrasing so downstream code doesn't care which
     dataset a chunk came from.
  3. Emit one Chunk per paragraph (chunk_type="text") and one per table row
     (chunk_type="table_row"), both tagged dataset="tatqa".
  4. Emit one Question per entry in `questions`, with gold_chunk_ids built from
     rel_paragraphs (map "order" -> paragraph uid/chunk_id) at minimum; table-evidence
     gold_chunk_ids can be left empty for v1 and revisited once B1 is working end-to-end
     (safety-net milestone first, per the brief's Month 3 plan).
  5. Mirror finqa_loader.py's CLI: --input, --split, --out-dir -> writes
     tatqa_{split}_chunks.jsonl / tatqa_{split}_questions.jsonl using the same schema.py
     dataclasses, so build_corpus.py (scripts/) doesn't need to know which loader ran.

A tiny 8-example fixture already exists at data/sample/tatqa_sample.json (pulled from the
real dev set) so you can unit-test this without downloading the full file first.
"""
raise NotImplementedError(
    "tatqa_loader.py is scaffolded but not implemented yet -- see the module docstring "
    "for the TODO list and the exact schema (verified against the real TAT-QA dev set)."
)
