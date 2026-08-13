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
    normalization, which is what TAT-QA span answers are graded against.

    The subtlety is that '.', '%' and '-' cannot simply be stripped: they carry meaning inside
    a number ("1.5", "4.35%", "-94"). But keeping them everywhere is worse, because a gold
    answer lifted from a document ends in a full stop and the prediction does not:

        gold  "lower level of R&D grants."   ->  [..., "grants."]
        pred  "lower level of R&D grants"    ->  [..., "grants"]

    which costs an exact match and a token of F1 on an answer that is word-for-word correct.
    So punctuation is stripped from the edges of *non-numeric* tokens only, and a trailing
    '.' is stripped from numeric ones ("173." -> "173", while "1.5" and "4.35%" survive).
    """
    s = str(text or "").lower()
    s = re.sub(r"[^\w\s.%-]", " ", s)
    tokens = []
    for token in s.split():
        if token in _ARTICLES:
            continue
        token = token.rstrip(".") if re.search(r"\d", token) else token.strip(".%-")
        if token:
            tokens.append(token)
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


def execution_match(pred, exe_answer, is_percent: bool = False, rel_tol: float = 0.01) -> bool:
    """FinQA's official metric: does the prediction equal the *executed* gold value.

    `exe_answer` is the result of running FinQA's gold program, so it carries full precision
    where the display string is rounded for presentation. Scoring against the display string
    marks a model wrong for being more accurate than the annotation -- a computed -6.8528
    fails a 1% tolerance against "-7%" -- which is why this exists as a separate relation.

    Percent handling is gated on `is_percent` rather than applied to every comparison.
    FinQA stores percentages as fractions and the prompt asks for percent form, so a factor
    of 100 is a unit convention *for those questions only*; accepting it everywhere would
    mark an answer that is wrong by 100x as correct.
    """
    if exe_answer is None:
        return False
    gold = normalize_number(exe_answer)
    if gold is None:                       # boolean gold ("yes"/"no") -- compare as text
        return exact_match(pred, exe_answer)
    p = normalize_number(pred)
    if p is None:
        return False
    if _close(p, gold, rel_tol):
        return True
    if not is_percent:
        return False
    # prediction in percent form against a fractional gold, or the reverse
    return _close(p, gold * 100, rel_tol) or (abs(p) <= 1 and _close(p * 100, gold, rel_tol))


def score_answer(pred, gold, answer_type: str | None = None, scale: str | None = None,
                 exe_answer=None, answer_is_percent: bool = False) -> dict:
    """Per-question answer scores. `is_numeric` records which metric is the meaningful one for
    this question, so the aggregate can report numeric accuracy over numeric questions rather
    than diluting it with span questions that were never going to parse as floats.

    When `exe_answer` is present (FinQA), it is authoritative for the numeric verdict and the
    display string is kept only for the text metrics. Both verdicts are recorded so the
    difference between them stays auditable rather than being a silent change of definition.
    """
    has_exe = exe_answer is not None and str(exe_answer).strip() != ""
    display_numeric = normalize_number(gold) is not None
    gold_is_numeric = display_numeric or (has_exe and normalize_number(exe_answer) is not None)

    display_match = numeric_match(pred, gold, scale=scale)
    exe_ok = execution_match(pred, exe_answer, answer_is_percent) if has_exe else None
    scores = {
        "is_numeric": 1.0 if gold_is_numeric else 0.0,
        "numeric_match": 1.0 if (exe_ok if exe_ok is not None else display_match) else 0.0,
        "display_match": 1.0 if display_match else 0.0,
        "exact_match": 1.0 if exact_match(pred, gold) else 0.0,
        "token_f1": round(token_f1(pred, gold), 4),
        "answer_type": answer_type,
    }
    if exe_ok is not None:
        scores["execution_match"] = 1.0 if exe_ok else 0.0
    return scores


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
