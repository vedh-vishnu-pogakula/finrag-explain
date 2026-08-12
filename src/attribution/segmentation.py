"""
Query segmentation: splitting a question into the units attribution is computed over.

This is the first real design decision of Contribution 1, and it decides what the explanation
can even say. Attributing over raw sub-word tokens produces explanations nobody can read
("##rev", "##enue"); attributing over the whole query produces one number and no explanation
at all. The unit here is the *readable span a financial analyst would point at*: a number, a
named entity, a content word.

Each unit carries a `kind`, and that is not cosmetic -- CLAUDE.md frames Contribution 1 as
"which query tokens/entities/numeric values drove the score", so the evaluation reports
attribution mass broken down by kind. "Numeric values carry N% of attribution mass on FinQA"
is a finding; "token 7 has weight 0.3" is not.

Units are stored as character spans into the original question, so a perturbation blanks a
span and leaves the surrounding text -- including punctuation -- exactly as written. Joining
surviving token strings with spaces instead would quietly change the query in ways unrelated
to the unit being tested, and the resulting score drop would be partly an artifact of that.

spaCy NER is used to group multi-word entities when the model is available, and the module
falls back to a capitalization heuristic when it isn't. The fallback is deliberate: the
pipeline must never hard-fail because an optional model wasn't downloaded, and the difference
only affects how units are *grouped*, never whether attribution runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Function words. Kept as attributable units -- they genuinely move a dense retriever's
# embedding, so dropping them from the analysis would silently misattribute their effect to
# neighbouring words. They're labelled so their mass can be reported separately.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "as", "that", "this", "these", "those", "it",
    "its", "what", "which", "who", "whom", "how", "when", "where", "why", "did", "do", "does",
    "has", "have", "had", "will", "would", "s",
}

# Numbers as an analyst writes them: 1,496.5 / $680 / 5.2% / 2019 / (774)
_NUMBER_RE = re.compile(r"\(?\$?\d[\d,]*\.?\d*%?\)?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'&./-]*")


@dataclass(frozen=True)
class QueryUnit:
    """One attributable span of the query."""
    index: int
    text: str
    start: int
    end: int
    kind: str      # "number" | "entity" | "term" | "stopword"

    def to_json(self) -> dict:
        return {"index": self.index, "text": self.text, "kind": self.kind}


def _spacy_entity_spans(question: str):
    """(start, end) spans of named entities, or [] if spaCy's model isn't present.

    Single-token entities ("Costco") are included as well as multi-word ones ("American
    Express"). Grouping only changes anything for the multi-word case, but claiming the
    single-token span relabels it from `term` to `entity`, and that distinction is what the
    mass-by-kind breakdown reports on. Numbers are claimed before entities in
    `segment_query`, so a DATE entity like "2019" stays a numeric unit.
    """
    try:
        import spacy
    except ImportError:
        return []
    global _NLP
    try:
        _NLP
    except NameError:
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
        except OSError:
            # Model not downloaded. Documented, non-fatal: run
            #   python -m spacy download en_core_web_sm
            # to enable entity grouping. Attribution runs identically without it.
            _NLP = None
    if _NLP is None:
        return []
    return [(e.start_char, e.end_char) for e in _NLP(question).ents]


def _fallback_entity_spans(question: str):
    """Runs of two or more capitalized words, ignoring a sentence-initial capital.

    A blunt stand-in for NER, but on financial questions it reliably catches the thing that
    matters -- company and segment names like "Transportation Solutions" -- which is the case
    where splitting into separate units would misattribute the entity's effect.
    """
    spans, run = [], []
    for m in _WORD_RE.finditer(question):
        word = m.group()
        if word[0].isupper() and m.start() > 0:
            run.append(m)
        else:
            if len(run) > 1:
                spans.append((run[0].start(), run[-1].end()))
            run = []
    if len(run) > 1:
        spans.append((run[0].start(), run[-1].end()))
    return spans


def segment_query(question: str, use_ner: bool = True) -> list[QueryUnit]:
    """Split `question` into ordered, non-overlapping attributable units.

    Precedence is numbers, then entities, then single words -- numbers first because a year
    inside an entity-looking span ("Fiscal 2019") should stay an independently testable
    numeric unit, which is the whole point of the numeric-attribution claim.
    """
    claimed = [False] * (len(question) + 1)
    spans: list[tuple[int, int, str]] = []

    def claim(start, end, kind):
        if any(claimed[start:end]):
            return
        for i in range(start, end):
            claimed[i] = True
        spans.append((start, end, kind))

    for m in _NUMBER_RE.finditer(question):
        if any(ch.isdigit() for ch in m.group()):
            claim(m.start(), m.end(), "number")

    entity_spans = (_spacy_entity_spans(question) if use_ner else []) \
        or (_fallback_entity_spans(question) if use_ner else [])
    for start, end in entity_spans:
        claim(start, end, "entity")

    for m in _WORD_RE.finditer(question):
        kind = "stopword" if m.group().lower() in STOPWORDS else "term"
        claim(m.start(), m.end(), kind)

    spans.sort()
    return [QueryUnit(index=i, text=question[s:e], start=s, end=e, kind=k)
            for i, (s, e, k) in enumerate(spans)]


def render(question: str, units: list[QueryUnit], mask) -> str:
    """Rebuild the query with only the units where `mask` is truthy.

    Removed spans are blanked in place rather than the survivors being re-joined, so word
    order, casing and punctuation of what remains are byte-identical to the original. Runs of
    whitespace are then collapsed -- without that, a removed unit leaves a double space, and
    the tokenizer sees a query that differs from the original in a second way.
    """
    kept = list(question)
    for unit, keep in zip(units, mask):
        if not keep:
            for i in range(unit.start, unit.end):
                kept[i] = " "
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def unit_kinds(units: list[QueryUnit]) -> dict:
    counts: dict = {}
    for u in units:
        counts[u.kind] = counts.get(u.kind, 0) + 1
    return counts
