"""
Answer-correctness metrics for baseline B1 (generation half).

Financial QA breaks naive exact match in ways that are entirely about formatting, not about
being wrong. Per the CLAUDE.md guardrail, numeric answers are normalized before comparison:

    "$1,496.5"  ==  "1496.50"  ==  "1,496.5"
    "(1,234)"   ==  "-1234"                     (accounting negative)
    "5.2%"      ==  "5.2"      ==  "0.052"      (only when a percent is actually in play)

and TAT-QA's `scale` field is carried through rather than dropped: an answer of `-94` with
`scale: "million"` means -94,000,000, so a prediction of "-94 million" and a prediction of
"-94" are both accepted, while a bare "-94000000" is accepted too.

Three numbers get reported per question, and they answer different questions:

  * `numeric_match` -- did the model get the *value* right, formatting aside. The headline for
    FinQA and for TAT-QA arithmetic/count questions.
  * `exact_match`   -- string equality after light normalization. Reported for span answers,
    where the expected output is text lifted from the document.
  * `token_f1`      -- partial credit on span answers (multi-span answers rarely match a gold
    string character-for-character even when they're right).

The separation matters for the project's failure taxonomy: a right-evidence-wrong-arithmetic
case shows up here as `numeric_match=0` with the gold chunk retrieved, which is a generation
failure, not a grounding failure. Keeping those apart is a working convention in CLAUDE.md,
not a nicety -- so the eval script logs retrieval and answer outcomes side by side.
"""
from __future__ import annotations

import re
from collections import Counter

# TAT-QA's `scale` field. "percent" is deliberately absent: it doesn't multiply the value, it
# says the number is already a percentage -- handled by the percent-equivalence rule instead.
SCALE_FACTORS = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
}

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_ARTICLES = {"a", "an", "the"}


def normalize_number(text) -> float | None:
    """Parse a financial number string to a float, or None if it isn't one.

    Handles currency symbols, thousands separators, percent signs, and the accounting
    convention of parenthesizing negatives. Returns None (rather than raising) for text
    answers, so callers can branch on it to decide which metric applies.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip()
    s = re.sub(r"[$€£¥,%\s]", "", s)
    s = s.rstrip(".")
    if not s or not re.fullmatch(r"-?\d*\.?\d+", s):
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def _close(a: float, b: float, rel_tol: float) -> bool:
    """Relative comparison with an absolute floor, so values near zero don't fail on a
    rounding difference that would be irrelevant at any other magnitude."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), 1e-6)


def numeric_match(pred, gold, scale: str | None = None, rel_tol: float = 0.01) -> bool:
    """True when prediction and gold are the same value.

    `rel_tol` of 1% absorbs the rounding the datasets themselves do (FinQA gold answers are
    frequently rounded to one decimal while the executed program isn't). Three relations are
    accepted beyond direct equality:

      - percent form: 5.2 vs 0.052, *only* when a "%" appears on either side or scale is
        "percent". Applying it unconditionally would make 5.2 and 0.052 equal for plain
        dollar amounts, which is a false positive, not a formatting difference.
      - scale form: gold 94 with scale "million" also matches a predicted 94,000,000.
      - sign-insensitive scale/percent forms are NOT accepted -- a sign error is a real error.
    """
    p, g = normalize_number(pred), normalize_number(gold)
    if p is None or g is None:
        return False
    if _close(p, g, rel_tol):
        return True

    is_percent = "%" in str(pred) or "%" in str(gold) or (scale or "").lower() == "percent"
    if is_percent and (_close(p, g * 100, rel_tol) or _close(p * 100, g, rel_tol)):
        return True

    factor = SCALE_FACTORS.get((scale or "").lower())
    if factor and (_close(p, g * factor, rel_tol) or _close(p * factor, g, rel_tol)):
        return True
    return False


def normalize_text(text) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace -- the standard SQuAD-style
    normalization, which is what TAT-QA span answers are graded against."""
    s = str(text or "").lower()
    s = re.sub(r"[^\w\s.%-]", " ", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred, gold) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def token_f1(pred, gold) -> float:
    """Partial credit for span answers. Multi-span gold answers are joined with ' | ' by the
    loaders, so a prediction listing the same spans in a different order still scores well."""
    p_tokens = normalize_text(pred).replace("|", " ").split()
    g_tokens = normalize_text(gold).replace("|", " ").split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_tokens)
    recall = n_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(pred, gold, answer_type: str | None = None, scale: str | None = None) -> dict:
    """Per-question answer scores. `is_numeric` records which metric is the meaningful one for
    this question, so the aggregate can report numeric accuracy over numeric questions rather
    than diluting it with span questions that were never going to parse as floats."""
    gold_is_numeric = normalize_number(gold) is not None
    return {
        "is_numeric": 1.0 if gold_is_numeric else 0.0,
        "numeric_match": 1.0 if numeric_match(pred, gold, scale=scale) else 0.0,
        "exact_match": 1.0 if exact_match(pred, gold) else 0.0,
        "token_f1": round(token_f1(pred, gold), 4),
        "answer_type": answer_type,
    }


def aggregate_answers(scores: list[dict]) -> dict:
    """Overall answer quality. Numeric accuracy is averaged over numeric questions only; span
    metrics over the rest. Averaging either over everything produces a number that means
    nothing on a mixed dataset like TAT-QA."""
    if not scores:
        return {}
    numeric = [s for s in scores if s["is_numeric"]]
    textual = [s for s in scores if not s["is_numeric"]]

    def mean(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None

    return {
        "n_questions": len(scores),
        "n_numeric": len(numeric),
        "n_textual": len(textual),
        "numeric_accuracy": mean(numeric, "numeric_match"),
        "span_exact_match": mean(textual, "exact_match"),
        "span_token_f1": mean(textual, "token_f1"),
        "overall_exact_match": mean(scores, "exact_match"),
    }
