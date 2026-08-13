"""
Sentence-level segmentation for the evidence-grounding layer.

Grounding asks a narrow question: *for each thing the answer asserts, which retrieved sentence
supports it?* That requires both sides split into units small enough to be checked
individually. A whole-answer-against-whole-evidence check answers "is this grounded" with one
number and no trail, which is exactly the thing this layer exists to replace.

Two asymmetries drive the design:

* **Answers here are short and often not prose.** B1 emits a bare value ("-42.4"), sometimes a
  span, occasionally a sentence. Running a sentence splitter over "-42.4" and getting one
  "claim" back is correct but useless, so a numeric or short-span answer is treated as a
  single claim and paired with the question to form a checkable proposition -- "-42.4" entails
  nothing on its own, but "the percentage change was -42.4" does. Without that, every numeric
  answer in FinQA would score as ungrounded no matter how good the evidence was.

* **Evidence is half prose, half linearized table rows.** A table row like
  `company the contingent rental of 2009 is 19 ; ... of 2007 is 33 ;` has no sentence
  boundaries a linguistic splitter can find -- spaCy returns it as one long sentence, which
  then entails almost nothing under an NLI model because the relevant fact is buried among
  four irrelevant ones. Table rows are therefore split on their own `;` delimiter, which is
  the structure `finqa_loader._linearize_table_row` put there in the first place.

spaCy is used for the prose path when available and degrades to a regex splitter when it is
not, matching how `src/attribution/segmentation.py` treats its NER dependency: the model is a
quality upgrade, not a hard requirement, so the tests stay offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Table rows are emitted with "; " between header->value pairs by both loaders. Splitting on
# that is not a linguistic decision, it is reading back the structure we wrote.
_ROW_DELIM_RE = re.compile(r"\s*;\s*")
# Prose fallback: split after . ! ? when followed by whitespace and a capital or digit. The
# lookbehind exclusion keeps common financial abbreviations and decimals from splitting.
_SENT_FALLBACK_RE = re.compile(r"(?<![A-Z])(?<!\d)[.!?]+\s+(?=[A-Z0-9])")

# Below this many characters an answer is a value or a short span, not a proposition, and is
# grounded as a single claim rather than sentence-split.
_SHORT_ANSWER_CHARS = 80

_NLP = None
_SPACY_TRIED = False


@dataclass
class Sentence:
    """One checkable unit of evidence, kept with the chunk it came from.

    `chunk_id` is what makes grounding actionable rather than descriptive: Contribution 2
    perturbs the evidence a metric claims to rely on, and it can only do that if a support
    decision points back at a specific retrieved chunk. `index` orders sentences within that
    chunk so a perturbation can remove one sentence rather than the whole chunk.
    """
    text: str
    chunk_id: str
    doc_id: str
    index: int
    is_table_row: bool = False

    def to_json(self) -> dict:
        return {
            "text": self.text, "chunk_id": self.chunk_id, "doc_id": self.doc_id,
            "index": self.index, "is_table_row": self.is_table_row,
        }


def _spacy():
    """Load spaCy's sentence splitter once, or return None and let the caller fall back.

    Only the parser-free sentencizer is needed, so the pipeline is stripped down -- loading
    NER and the parser to split sentences would cost far more than it returns here.
    """
    global _NLP, _SPACY_TRIED
    if _SPACY_TRIED:
        return _NLP
    _SPACY_TRIED = True
    try:
        import spacy

        try:
            _NLP = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "textcat"])
        except OSError:
            # No model installed: a blank pipeline with a rule-based sentencizer still splits
            # prose correctly and needs no download.
            #   python -m spacy download en_core_web_sm
            _NLP = spacy.blank("en")
            _NLP.add_pipe("sentencizer")
    except ImportError:
        _NLP = None
    return _NLP


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, preferring spaCy and falling back to a regex."""
    text = (text or "").strip()
    if not text:
        return []
    nlp = _spacy()
    if nlp is not None:
        parts = [s.text.strip() for s in nlp(text).sents]
    else:
        parts = [p.strip() for p in _SENT_FALLBACK_RE.split(text)]
    return [p for p in parts if p]


def split_table_row(text: str) -> list[str]:
    """Split a linearized table row into its header->value facts.

    Each fact keeps the row label, because "the contingent rental of 2007 is 33" is checkable
    on its own while "of 2007 is 33" is not. The label is whatever precedes the first fact --
    both loaders put it there.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip(" ;") for p in _ROW_DELIM_RE.split(text) if p.strip(" ;")]
    return parts or [text]


def evidence_sentences(retrieved) -> list[Sentence]:
    """Flatten retrieved chunks into individually checkable sentences.

    Accepts either RetrievedChunk objects or the plain dicts stored in the B1 checkpoints, so
    grounding can be run live or re-run over saved results without a re-generation pass -- the
    same property that makes `run_b1_rag.py --rescore` free.
    """
    out: list[Sentence] = []
    for hit in retrieved:
        chunk = getattr(hit, "chunk", None)
        if chunk is not None:
            text, chunk_id = chunk.text, chunk.chunk_id
            doc_id, chunk_type = chunk.doc_id, chunk.chunk_type
        else:
            text, chunk_id = hit.get("text", ""), hit.get("chunk_id", "")
            doc_id, chunk_type = hit.get("doc_id", ""), hit.get("chunk_type", "")
        is_row = chunk_type == "table_row"
        parts = split_table_row(text) if is_row else split_sentences(text)
        for i, part in enumerate(parts):
            out.append(Sentence(text=part, chunk_id=chunk_id, doc_id=doc_id,
                                index=i, is_table_row=is_row))
    return out


@dataclass
class Claim:
    """One checkable assertion extracted from an answer.

    `kind` decides how it gets verified, and the split is not a convenience -- the two kinds
    are not checkable by the same instrument:

      * `numeric` -- an operand the answer's arithmetic consumed. Verified by *provenance*:
        does this figure occur in the retrieved evidence, and in which sentence.
      * `text` -- a span or sentence the answer asserts. Verified by NLI entailment.
    """
    text: str
    kind: str = "text"
    value: float | None = None


def answer_claims(answer: str, question: str | None = None,
                  expression: str | None = None) -> list[Claim]:
    """Split a generated answer into individually checkable claims.

    **Numeric answers are grounded on their operands, not on the final value.** This is the
    central design decision of the grounding layer, and it was forced by measurement: running
    NLI on the final value scores ~0.09 entailment even when the evidence is exactly right,
    because "the contingent rental of 2009 is 19" genuinely does not entail "the change was
    -42.4" -- getting from one to the other requires arithmetic, which no NLI model performs.
    Reporting those answers as ungrounded would be an artefact of the instrument, not a
    property of the system.

    Program-of-thought makes the better check available for free. `answer_expression` names
    the exact figures the answer consumed, so each operand becomes a claim whose truth is a
    matter of provenance: is this number actually in the retrieved evidence, and where. That
    is deterministic, needs no model, and -- crucially for Contribution 2 -- yields a concrete
    (chunk, sentence) target to perturb.

    Text answers keep the NLI path, where entailment is the right instrument. A short text
    answer is folded together with the question, because "Automotive" entails nothing on its
    own while "which segment grew fastest? Automotive" is a proposition.
    """
    answer = (answer or "").strip()
    if not answer:
        return []

    operands = expression_operands(expression)
    if operands:
        return [Claim(text=_format_operand(v), kind="numeric", value=v) for v in operands]

    if len(answer) <= _SHORT_ANSWER_CHARS:
        question = (question or "").strip()
        return [Claim(text=f"{question} {answer}".strip() if question else answer)]
    return [Claim(text=s) for s in split_sentences(answer)] or [Claim(text=answer)]


# Unsigned on purpose, and for the same reason as in tatqa_loader: in "680-774" the '-' is the
# subtraction operator, not the sign of the operand. Provenance is matched on magnitude.
_OPERAND_RE = re.compile(r"\d[\d,]*\.?\d*")

# Constants a model writes as scaffolding rather than as figures read out of the document.
# Grounding them would manufacture support -- "100" is in every percentage expression and in
# plenty of tables, so counting it as evidence-backed inflates the grounding rate for free.
_SCAFFOLD_CONSTANTS = {100.0, 1.0, 0.0, 2.0, 1000.0, 12.0, 365.0}


def _format_operand(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def expression_operands(expression: str | None) -> list[float]:
    """The distinct figures an arithmetic expression consumed, in order of first appearance."""
    if not expression:
        return []
    values = []
    for match in _OPERAND_RE.finditer(str(expression)):
        try:
            value = abs(float(match.group().replace(",", "")))
        except ValueError:
            continue
        if value not in _SCAFFOLD_CONSTANTS and value not in values:
            values.append(value)
    return values
