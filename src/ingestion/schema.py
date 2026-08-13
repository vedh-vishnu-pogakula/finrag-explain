"""
Shared internal schema that every dataset loader (finqa_loader.py, tatqa_loader.py, ...)
must output. Downstream code (retrieval, attribution, grounding) only ever talks to these
two dataclasses -- it never sees FinQA's or TAT-QA's native JSON shapes. This is what lets
Contribution 1/2 code stay dataset-agnostic even though the raw formats are quite different.

Per CLAUDE.md: don't force one shared *parser* onto both datasets -- but they DO share this
one output schema. Two loaders, one contract.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
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
    # TAT-QA: thousand | million | billion | percent | "" -- changes what a bare number means
    # ("-94" with scale "million" is -94,000,000). Carried through to answer scoring; dropping
    # it silently makes correct numeric answers look wrong.
    scale: Optional[str] = None
    # FinQA ships two gold answers and they are not interchangeable. `answer` is the
    # human-readable display string, rounded for presentation ("-7%", "11%"); `exe_answer` is
    # the executed value of the gold program (-0.06853, 0.10745). FinQA's own evaluation is
    # *execution accuracy* against the latter, and scoring against the former silently marks
    # exact answers wrong -- a model that computes -6.8528 fails a 1% tolerance against "-7%".
    # Stored as a string so the jsonl round-trip stays lossless and boolean answers ("yes")
    # survive; the metrics parse it. None for TAT-QA, which has no equivalent field.
    exe_answer: Optional[str] = None
    # True when the gold answer is a percentage. FinQA stores those as fractions while the
    # prompt asks the model for percent form (42.4, not 0.424), so scoring has to know that
    # a factor of 100 between prediction and gold is a unit convention rather than an error.
    # Applying that relation unconditionally would mark genuinely wrong answers correct.
    answer_is_percent: bool = False

    def to_json(self) -> dict:
        return asdict(self)


# ---- jsonl round-tripping -------------------------------------------------------------
# The loaders write these files; retrieval/attribution/grounding read them back. Keeping both
# halves here is what lets downstream code import only schema.py and stay dataset-agnostic --
# it never needs to know whether a chunk came out of finqa_loader or tatqa_loader.

def write_jsonl(items, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item.to_json()) + "\n")
    return path


def read_chunks(path) -> list:
    with open(path) as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def read_questions(path) -> list:
    with open(path) as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]
