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
import re
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
    """Returns (chunks, row_cells) where row_cells maps chunk_id -> the row's normalized cell
    values. Gold-evidence matching needs cell-level values, not the linearized string: a
    substring test on the linearized text matches "680" inside "6,801"."""
    table = table_obj.get("table", [])
    if not table:
        return [], {}
    n_cols = max(len(r) for r in table)
    header_rows, data_rows = _split_header_and_data_rows(table)
    col_headers = _merge_header_cells(header_rows, n_cols) if header_rows else [""] * n_cols

    chunks, row_cells = [], {}
    section_label = None
    for i, row in enumerate(data_rows):
        rest = row[1:]
        is_section_label = row[0].strip() and not any(c.strip() for c in rest)
        if is_section_label:
            section_label = row[0].strip()
            continue  # not independently retrievable -- it qualifies the rows beneath it
        if not any(c.strip() for c in row):
            continue
        chunk_id = f"table_row_{i}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            dataset="tatqa",
            chunk_type="table_row",
            text=_linearize_table_row(col_headers, row, section_label),
            table_row_index=i,
        ))
        row_cells[chunk_id] = [v for v in (_normalize_cell(c) for c in row) if v is not None]
    return chunks, row_cells


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


# Unsigned on purpose: in a derivation the '-' in "680-774" is the subtraction operator, not
# the sign of the operand. Signs are handled by comparing magnitudes in _table_gold_ids.
_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def _normalize_cell(value):
    """Table cells and answer strings into one comparable space: floats where the text is a
    number (so "$1,496.5", "1496.50" and "(1,496.5)" -> 1496.5 / -1496.5), lowercase strings
    otherwise. Returns None for blanks. Mirrors the numeric-normalization guardrail in
    CLAUDE.md -- currency symbols, thousands separators and percent signs must not make two
    identical values look different."""
    s = str(value).strip()
    if not s:
        return None
    negative = s.startswith("(") and s.endswith(")")
    stripped = s.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        f = float(stripped)
        return -f if negative else f
    except ValueError:
        return s.lower()


def _derivation_operands(derivation):
    """Values cited in a question's `derivation` -- the cells the annotator actually read out
    of the table, which is what makes them usable as evidence pointers when the answer itself
    appears in no row. Two shapes occur:

      arithmetic: "680-774" -> [680.0, 774.0];  "(5,686-6,092)/6,092" -> [5686.0, 6092.0]
      count:      "Customer relationships## Underlying rights" -> those two row labels
                  (the answer is "2", which is nowhere in the table)
    """
    if not derivation:
        return []
    text = str(derivation)
    values = []
    if "##" in text:
        values += [p.strip().lower() for p in text.split("##") if p.strip()]
    for token in _NUM_RE.findall(text):
        v = _normalize_cell(token)
        if isinstance(v, float):
            values.append(v)
    # "* 100" / "/ 100" in a percentage derivation is a unit conversion, not a cell that was
    # read out of the table -- keeping it would mark any row containing 100 as gold evidence.
    # Cost: a genuine 100-valued operand in a percentage derivation is dropped too (rare).
    if re.search(r"[*/]\s*100(?!\d)", text):
        values = [v for v in values if v != 100.0]
    return list(dict.fromkeys(values))  # dedupe, keep order ("(a-b)/b" cites b twice)


def _table_gold_ids(row_cells, answers, derivation=None):
    """Best-effort table-evidence pointers. TAT-QA has no gold row index (per CLAUDE.md: no
    ground truth for 'correct attribution' either -- treated as expected, not papered over),
    so a row counts as evidence when one of its *cells* matches either

      - an answer value (span / multi-span / count answers are lifted from cells), or
      - a numeric operand of the question's `derivation` (arithmetic answers are *computed*,
        so the answer itself appears in no row -- the operands do).

    Matching is cell-level, not substring-over-the-linearized-row: "680" must not match the
    cell "6,801". Non-numeric answers do allow substring containment, since a span answer can
    be part of a longer row label ("Appliances" in "Appliances, net"). Numbers match on
    magnitude, because the two sides use different sign conventions for the same figure: a
    table shows a negative as "(774)" while the derivation that consumed it writes "774".

    Still a heuristic and still not a ground-truth label -- a value appearing in several rows
    marks all of them. Don't feed these to eval code without re-reading this docstring.
    """
    targets = set()
    for a in answers:
        v = _normalize_cell(a)
        if v is not None and v != "":
            targets.add(v)
    targets.update(_derivation_operands(derivation))
    if not targets:
        return []

    numeric = {abs(t) for t in targets if isinstance(t, float)}
    textual = {t for t in targets if isinstance(t, str)}

    ids = []
    for chunk_id, cells in row_cells.items():
        cell_nums = {abs(c) for c in cells if isinstance(c, float)}
        cell_strs = [c for c in cells if isinstance(c, str)]
        hit = bool(cell_nums & numeric) or any(
            t == c or (len(t) > 2 and t in c) for t in textual for c in cell_strs
        )
        if hit:
            ids.append(chunk_id)
    return ids


def load_tatqa(path: str):
    """Yields (chunks, question) tuples, one per question -- chunks are the full doc's chunk
    set (paragraphs + table rows), repeated per question in the same doc; dedupe with
    build_chunk_index() (same helper contract as finqa_loader.py)."""
    raw = json.loads(Path(path).read_text())

    for doc in raw:
        doc_id = doc["table"]["uid"]
        table_chunks, row_cells = _table_chunks(doc["table"], doc_id)
        text_chunks = _text_chunks(doc.get("paragraphs", []), doc_id)
        chunks = text_chunks + table_chunks
        para_chunk_ids = {c.chunk_id for c in text_chunks}

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
                gold_ids += _table_gold_ids(row_cells, answers, q.get("derivation"))

            question = Question(
                qa_id=q["uid"],
                doc_id=doc_id,
                dataset="tatqa",
                question=q.get("question", ""),
                answer=" | ".join(str(a) for a in answers),
                gold_chunk_ids=gold_ids,
                answer_type=q.get("answer_type"),
                scale=q.get("scale") or None,
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
