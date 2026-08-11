"""
TAT-QA loader.

Source: Zhu et al. 2021 (arxiv.org/abs/2105.07624), official data at
https://github.com/NExTplusplus/TAT-QA/tree/master/dataset_raw
(tatqa_dataset_train.json / _dev.json / _test.json -- test.json answers are withheld,
dev.json is this project's eval split per the guardrail in CLAUDE.md).

Native shape (one dict per report excerpt -- confirmed against the real dev set):
    {
      "table": {"uid": "...", "table": [[row0...], [row1...], ...]},
      "paragraphs": [{"uid": "...", "order": 1, "text": "..."}, ...],
      "questions": [
          {"uid": "...", "question": "...", "answer": [...], "answer_type": "span"|...,
           "answer_from": "table"|"text"|"table-text", "rel_paragraphs": ["2"], "scale": "..."},
          ...
      ]
    }

This is intentionally a SEPARATE parser from finqa_loader.py (not a shared one) -- confirmed
by inspecting real examples, TAT-QA tables routinely have 2-3 leading header/annotation rows
(e.g. a merged date-range row, a year row, a units row like "U.S. $ in thousands") and interior
"section label" rows with a category name and no values (e.g. "Assets", "Transportation
Solutions:") that apply to the data rows beneath them until the next section label. FinQA never
has either of these. Forcing one parser onto both would make the header/section handling below
unreadable.
"""
import json
import argparse
from pathlib import Path

from schema import Chunk, Question

def _split_header_and_data_rows(table):
    """Header rows are the leading contiguous rows whose first (row-label) column is blank.
    This is deliberately NOT a numeric-cell check -- an earlier version tried "does this row
    contain numeric-looking cells" and misclassified year-label rows ('2019', '2018') as data
    rows, since bare years match a numeric pattern just as well as dollar figures do. Row-label
    presence is more reliable across the real dev set: every header/annotation row we inspected
    (date ranges, year labels, units-in-thousands notes) leaves column 0 blank; every data row
    (and every section-label row, handled separately below) has something in column 0.
    Per CLAUDE.md guardrail: don't assume a single clean header row -- this handles the common
    2-3 row header case seen in the real data; genuinely irregular tables may still need a
    manual look."""
    header_rows, data_start = [], len(table)
    for i, row in enumerate(table):
        if row[0].strip():
            data_start = i
            break
        header_rows.append(row)
    return header_rows, table[data_start:]


def _merge_header_cells(header_rows, n_cols):
    """Concatenate header rows into one label per column, forward-filling blanks (a blank
    header cell means 'same as the nearest non-blank cell to the left on that row' -- the
    common convention for merged spreadsheet cells)."""
    merged = [""] * n_cols
    for row in header_rows:
        filled = list(row) + [""] * (n_cols - len(row))
        last = ""
        for i in range(n_cols):
            cell = filled[i].strip()
            if cell:
                last = cell
            elif last and i > 0:
                cell = last  # forward-fill a merged blank cell
            if cell:
                merged[i] = f"{merged[i]} {cell}".strip() if merged[i] else cell
    return merged


def _linearize_table_row(col_headers, row, section_label, row_label_col=0):
    row_label = row[row_label_col].strip()
    if section_label:
        row_label = f"{section_label} — {row_label}" if row_label else section_label
    parts = []
    for col_idx, value in enumerate(row):
        if col_idx == row_label_col or not str(value).strip():
            continue
        header = col_headers[col_idx] if col_idx < len(col_headers) else ""
        if not header:
            continue
        parts.append(f"the {row_label} of {header} is {value} ;")
    return f"company {' '.join(parts)}" if parts else row_label


def _table_chunks(table_obj, doc_id):
    table = table_obj.get("table", [])
    if not table:
        return []
    n_cols = max(len(r) for r in table)
    header_rows, data_rows = _split_header_and_data_rows(table)
    col_headers = _merge_header_cells(header_rows, n_cols) if header_rows else [""] * n_cols

    chunks = []
    section_label = None
    for i, row in enumerate(data_rows):
        rest = row[1:]
        is_section_label = row[0].strip() and not any(c.strip() for c in rest)
        if is_section_label:
            section_label = row[0].strip()
            continue  # not independently retrievable -- it qualifies the rows beneath it
        if not any(c.strip() for c in row):
            continue
        chunks.append(Chunk(
            chunk_id=f"table_row_{i}",
            doc_id=doc_id,
            dataset="tatqa",
            chunk_type="table_row",
            text=_linearize_table_row(col_headers, row, section_label),
            table_row_index=i,
        ))
    return chunks


def _text_chunks(paragraphs, doc_id):
    chunks = []
    for p in paragraphs:
        text = p.get("text", "").strip()
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=f"para_{p['order']}",
            doc_id=doc_id,
            dataset="tatqa",
            chunk_type="text",
            text=text,
        ))
    return chunks


def _table_gold_ids(table_chunks, answers):
    """No direct gold row index exists for table evidence in TAT-QA (per CLAUDE.md: no ground
    truth for 'correct attribution' either -- this project treats that as expected, not a gap
    to silently paper over). Best-effort v1: a table-row chunk counts as gold evidence if its
    linearized text contains one of the answer strings verbatim. This is a heuristic, not a
    ground-truth label -- documented here so it isn't mistaken for one downstream."""
    ids = []
    for chunk in table_chunks:
        if any(str(a).strip() and str(a).strip() in chunk.text for a in answers):
            ids.append(chunk.chunk_id)
    return ids


def load_tatqa(path: str):
    """Yields (chunks, question) tuples, one per question -- chunks are the full doc's chunk
    set (paragraphs + table rows), repeated per question in the same doc; dedupe with
    build_chunk_index() (same helper contract as finqa_loader.py)."""
    raw = json.loads(Path(path).read_text())

    for doc in raw:
        doc_id = doc["table"]["uid"]
        chunks = _text_chunks(doc.get("paragraphs", []), doc_id) + _table_chunks(doc["table"], doc_id)
        table_chunks = [c for c in chunks if c.chunk_type == "table_row"]
        para_chunk_ids = {c.chunk_id for c in chunks if c.chunk_type == "text"}

        for q in doc.get("questions", []):
            answers = q.get("answer", [])
            if not isinstance(answers, list):
                answers = [answers]

            gold_ids = []
            for order in q.get("rel_paragraphs", []):
                cid = f"para_{order}"
                if cid in para_chunk_ids:
                    gold_ids.append(cid)
            if q.get("answer_from") in ("table", "table-text"):
                gold_ids += _table_gold_ids(table_chunks, answers)

            question = Question(
                qa_id=q["uid"],
                doc_id=doc_id,
                dataset="tatqa",
                question=q.get("question", ""),
                answer=" | ".join(str(a) for a in answers),
                gold_chunk_ids=gold_ids,
                answer_type=q.get("answer_type"),
            )
            yield chunks, question


def build_chunk_index(all_chunks):
    seen = {}
    for c in all_chunks:
        seen[(c.doc_id, c.chunk_id)] = c
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser(description="Load TAT-QA -> chunks.jsonl + questions.jsonl")
    ap.add_argument("--input", required=True, help="path to TAT-QA train/dev json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", default="../../data/processed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_chunks, all_questions = [], []
    for chunks, question in load_tatqa(args.input):
        all_chunks.extend(chunks)
        all_questions.append(question)

    deduped = build_chunk_index(all_chunks)

    chunks_path = out_dir / f"tatqa_{args.split}_chunks.jsonl"
    questions_path = out_dir / f"tatqa_{args.split}_questions.jsonl"

    with open(chunks_path, "w") as f:
        for c in deduped:
            f.write(json.dumps(c.to_json()) + "\n")
    with open(questions_path, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q.to_json()) + "\n")

    n_with_gold = sum(1 for q in all_questions if q.gold_chunk_ids)
    print(f"[tatqa_loader] {len(all_questions)} questions, {len(deduped)} unique chunks "
          f"across {len(set(c.doc_id for c in deduped))} documents")
    print(f"[tatqa_loader] {n_with_gold}/{len(all_questions)} questions have >=1 gold_chunk_id "
          f"(table-evidence gold ids are heuristic -- see _table_gold_ids docstring)")
    print(f"[tatqa_loader] wrote {chunks_path}")
    print(f"[tatqa_loader] wrote {questions_path}")


if __name__ == "__main__":
    main()
