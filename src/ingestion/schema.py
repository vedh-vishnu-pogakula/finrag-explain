"""
Shared internal schema that every dataset loader (finqa_loader.py, tatqa_loader.py, ...)
must output. Downstream code (retrieval, attribution, grounding) only ever talks to these
two dataclasses -- it never sees FinQA's or TAT-QA's native JSON shapes. This is what lets
Contribution 1/2 code stay dataset-agnostic even though the raw formats are quite different.

Per CLAUDE.md: don't force one shared *parser* onto both datasets -- but they DO share this
one output schema. Two loaders, one contract.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Chunk:
    """One retrievable unit: either a sentence (text) or a table row, already linearized
    to plain text so the embedding model never has to special-case tables."""
    chunk_id: str          # e.g. "text_43" or "table_3" -- stable, human-readable
    doc_id: str             # groups chunks that came from the same source document/report
    dataset: str             # "finqa" | "tatqa"
    chunk_type: str          # "text" | "table_row"
    text: str                 # linearized text actually fed to the embedding model
    source_file: Optional[str] = None
    table_row_index: Optional[int] = None   # set only when chunk_type == "table_row"

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Question:
    """One QA pair, with whatever gold evidence pointers the source dataset provides.
    gold_chunk_ids should line up 1:1 with Chunk.chunk_id values from the same doc_id --
    that's what makes Precision@k/Recall@k for retrieval (and later, attribution
    validation) possible without inventing a new labeling scheme."""
    qa_id: str
    doc_id: str
    dataset: str
    question: str
    answer: str
    gold_chunk_ids: list = field(default_factory=list)
    program: Optional[str] = None     # FinQA-style reasoning program, if present
    answer_type: Optional[str] = None  # TAT-QA: span | arithmetic | ... (None for FinQA)

    def to_json(self) -> dict:
        return asdict(self)
