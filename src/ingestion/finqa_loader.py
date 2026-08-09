"""
FinQA loader.

Source: Chen et al. 2021 (arxiv.org/abs/2109.00122), official data at
https://github.com/czyssrs/FinQA/tree/main/dataset (train.json / dev.json / test.json).

Native shape (one dict per financial report excerpt):
    {
      "id": "V/2008/page_17.pdf-1",
      "pre_text":  [sentence, sentence, ...],
      "table":     [[header_col0, header_col1, ...], [row1_col0, row1_col1, ...], ...],
      "post_text": [sentence, sentence, ...],
      "qa": {
          "question": "...", "answer": "...", "exe_ans": 127.4, "program": "divide(637, const_5)",
          "gold_inds": {"table_3": "company the american express of ... is 637 ; ...", ...}
      }
    }

Indexing convention FinQA itself uses (we preserve it so gold_inds line up with our chunk_ids
for free, with zero re-mapping):
  - "text_N"  -> the N-th sentence in (pre_text + post_text), 0-indexed, continuous across the two.
  - "table_N" -> row N of `table` (row 0 is the header row itself; data rows start at 1).

We linearize each table row as "<row_label> the <row_label> of <col_header> is <value> ; ..."
which is exactly the format FinQA's own gold_inds already use -- so our chunks are directly
diffable against gold evidence, and we are not "flattening the whole table into one string"
(the guardrail this project explicitly needs to avoid): each row is its own chunk, and every
value is still tied to its column header.
"""
import json
import argparse
from pathlib import Path

from schema import Chunk, Question


def _linearize_table_row(header_row, row, row_label_col=0):
    """Turn one data row into 'the <row_label> of <col> is <val> ; ...', matching FinQA's
    own gold_inds phrasing so retrieval chunks are directly comparable to gold evidence."""
    row_label = row[row_label_col].strip()
    parts = []
    for col_idx, value in enumerate(row):
        if col_idx == row_label_col:
            continue
        header = header_row[col_idx].strip()
        if not header or not str(value).strip():
            continue
        parts.append(f"the {row_label} of {header} is {value} ;")
    return f"company {' '.join(parts)}" if parts else f"{row_label}"


def load_finqa(path: str, dataset_split: str = "train"):
    """Yields (chunks, question) tuples: one Chunk list + one Question per source example.
    Chunks for the same doc_id repeat across examples that share a source document in FinQA's
    raw files (that's fine -- dedupe by chunk_id when building the final chunks.jsonl, see
    build_chunk_index() below)."""
    raw = json.loads(Path(path).read_text())

    for ex in raw:
        doc_id = ex["id"]
        chunks = []

        sentences = list(ex.get("pre_text", [])) + list(ex.get("post_text", []))
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            chunks.append(Chunk(
                chunk_id=f"text_{i}",
                doc_id=doc_id,
                dataset="finqa",
                chunk_type="text",
                text=sent,
                source_file=ex.get("filename"),
            ))

        table = ex.get("table", [])
        if table:
            header_row = table[0]
            for row_idx, row in enumerate(table):
                if row_idx == 0:
                    continue  # header row itself isn't a retrievable chunk
                chunks.append(Chunk(
                    chunk_id=f"table_{row_idx}",
                    doc_id=doc_id,
                    dataset="finqa",
                    chunk_type="table_row",
                    text=_linearize_table_row(header_row, row),
                    source_file=ex.get("filename"),
                    table_row_index=row_idx,
                ))

        qa = ex.get("qa", {})
        question = Question(
            qa_id=f"{doc_id}::q0",
            doc_id=doc_id,
            dataset="finqa",
            question=qa.get("question", ""),
            answer=str(qa.get("answer", qa.get("exe_ans", ""))),
            gold_chunk_ids=list(qa.get("gold_inds", {}).keys()),
            program=qa.get("program"),
        )

        yield chunks, question


def build_chunk_index(all_chunks):
    """Dedupe chunks by (doc_id, chunk_id) -- multiple questions in the same raw file can
    point at the same source document."""
    seen = {}
    for c in all_chunks:
        seen[(c.doc_id, c.chunk_id)] = c
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser(description="Load FinQA -> chunks.jsonl + questions.jsonl")
    ap.add_argument("--input", required=True, help="path to FinQA train/dev/test.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", default="../../data/processed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_chunks, all_questions = [], []
    for chunks, question in load_finqa(args.input, args.split):
        all_chunks.extend(chunks)
        all_questions.append(question)

    deduped = build_chunk_index(all_chunks)

    chunks_path = out_dir / f"finqa_{args.split}_chunks.jsonl"
    questions_path = out_dir / f"finqa_{args.split}_questions.jsonl"

    with open(chunks_path, "w") as f:
        for c in deduped:
            f.write(json.dumps(c.to_json()) + "\n")

    with open(questions_path, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q.to_json()) + "\n")

    print(f"[finqa_loader] {len(all_questions)} questions, {len(deduped)} unique chunks "
          f"across {len(set(c.doc_id for c in deduped))} documents")
    print(f"[finqa_loader] wrote {chunks_path}")
    print(f"[finqa_loader] wrote {questions_path}")


if __name__ == "__main__":
    main()
