"""
A safe arithmetic evaluator for program-of-thought generation.

Why this module exists is the single most important thing to understand about B1's second
iteration. The first iteration asked a 1.5B model to read a table, decide which two figures
mattered, do the arithmetic in its head, and report only the final value. It scored 4.6% on
FinQA. The failure breakdown said why: of 250 questions, 11 were retrieval failures and 227
were generation failures -- and inside those 227, 24% of answers were a number copied straight
out of the evidence (no arithmetic attempted at all) while 19% were *unparseable* because the
model had written out its working, e.g.

    "$135.02 - $148.92 = -$13.90"

which is the correct computation, thrown away by a metric expecting a bare number.

That is not a knowledge failure, it is a division-in-the-head failure plus an output-format
failure. Both disappear if the model is asked for the *expression* and the expression is
executed here, in Python, exactly. This is the standard program-of-thought / FinQANet
formulation for FinQA -- the dataset ships a gold `program` field precisely because its authors
expected systems to emit programs rather than final values -- so it is a correct baseline
construction, not a trick. Retrieval quality, the model, and the evidence-bound constraint are
all unchanged; only the output channel moves from "a number" to "a number *or* the arithmetic
that produces it".

Security: the model's output is untrusted text, so `eval()` is not an option -- the whole point
of walking an AST with an explicit operator whitelist is that `__import__("os").system(...)`
cannot be expressed at all. Only literals and the five arithmetic operators survive the walk.
Names, calls, attributes, subscripts and comprehensions all raise.

Guards worth keeping:

* **Division by zero returns None, not inf.** A model that writes `x / 0` has made a reasoning
  error, and scoring `inf` against a gold value would silently count it as merely wrong rather
  than as malformed -- the distinction matters for the failure taxonomy.
* **Magnitude and length caps.** `9**9**9` is a one-line denial of service on any evaluator
  that allows `**`; the exponent cap makes it a rejection instead of a hang.
"""
from __future__ import annotations

import ast
import operator
import re

# Explicit whitelist. Anything not in this dict is a rejection, which is what makes the
# evaluator safe against arbitrary model output rather than merely unlikely to break.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Caps chosen to be far outside any real financial figure while still stopping a pathological
# expression from consuming the run. FinQA's largest values are in the billions.
_MAX_EXPR_CHARS = 200
_MAX_ABS_EXPONENT = 8

# FinQA's own program DSL, which the model sometimes imitates because the format is all over
# the web. Translating it rescues answers that are otherwise correct.
_DSL_BINARY = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "exp": "**"}
_CALL_START_RE = re.compile(r"\b(add|subtract|multiply|divide|exp)\s*\(")
# FinQA writes literals like `const_100` and `const_1000` in its programs.
_CONST_RE = re.compile(r"\bconst_(\d+(?:\.\d+)?)\b")
# A percent literal ("12%", "4.35%") is a *value*, not a calculation. Matching the full run of
# digits matters: an expression anchored on a single digit turns "12%" into "1(2/100)", which
# is a syntax error rather than a number.
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _match_paren(text: str, open_idx: int) -> int | None:
    """Index of the ')' closing the '(' at `open_idx`, or None if it is unbalanced."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _split_args(text: str) -> list[str]:
    """Split on top-level commas only, so nested calls survive as single arguments."""
    args, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(text[start:i])
            start = i + 1
    args.append(text[start:])
    return [a.strip() for a in args]


def _expand_dsl(text: str) -> str:
    """Rewrite `divide(subtract(a, b), c)` into `((a - b) / c)`.

    Recursive rather than regex-substituted: a regex that matches balanced parentheses either
    fails on the nested case (which is the common one -- FinQA's multi-step programs are all
    nested) or over-matches across sibling calls.
    """
    match = _CALL_START_RE.search(text)
    if match is None:
        return text
    open_idx = text.index("(", match.end() - 1)
    close_idx = _match_paren(text, open_idx)
    if close_idx is None:
        return text
    args = _split_args(text[open_idx + 1:close_idx])
    if len(args) != 2:
        # Unary or table-level DSL ops (table_sum, table_average) have no arithmetic
        # translation. Leaving the call intact makes it a parse failure downstream, which is
        # the honest outcome -- better than inventing a value for an operation we didn't run.
        return text
    body = f"({_expand_dsl(args[0])} {_DSL_BINARY[match.group(1)]} {_expand_dsl(args[1])})"
    return text[:match.start()] + body + _expand_dsl(text[close_idx + 1:])


class CalculatorError(ValueError):
    """Raised for anything that is not a plain arithmetic expression."""


def _clean(expression: str, expand_percent: bool = True) -> str:
    """Strip financial formatting so a model-written expression parses as arithmetic.

    Currency symbols, thousands separators and stray whitespace carry no arithmetic meaning.
    Percent literals do: "12%" is 12/100, and rewriting it that way is what lets an expression
    like `1500 * 12%` evaluate instead of failing on a modulo of a float.

    `expand_percent=False` returns the text with percent literals left alone, which is what
    `is_program` needs -- the rewrite introduces a `/`, and a caller looking for an operator to
    decide "is this a calculation?" would otherwise classify the bare value "4.35%" as one and
    execute it down to 0.0435, turning a correct answer into a wrong one.

    An `=` is treated as end-of-expression. Models routinely write `(a - b) / b = 0.42`; the
    left side is the program, the right side is the model's own (often wrong) arithmetic, so
    keeping the left side and discarding the right is exactly the repair that matters.
    """
    text = str(expression or "").strip()
    # Models wrap expressions in markdown or prose far more often than they should.
    text = text.strip("`").replace("\\", "")
    if text.lower().startswith("answer"):
        text = text.split(":", 1)[-1]
    if "=" in text:
        text = text.split("=", 1)[0]

    text = _CONST_RE.sub(r"\1", text)
    text = _expand_dsl(text)
    text = re.sub(r"[$€£¥,]", "", text)
    if expand_percent:
        text = _PERCENT_RE.sub(r"(\1/100)", text)
    return text.strip()


def _walk(node) -> float:
    """Recursively evaluate a whitelisted arithmetic AST."""
    if isinstance(node, ast.Expression):
        return _walk(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"non-numeric literal {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unsupported unary operator {type(node.op).__name__}")
        return op(_walk(node.operand))
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unsupported operator {type(node.op).__name__}")
        left, right = _walk(node.left), _walk(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise CalculatorError("division by zero")
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_ABS_EXPONENT:
            raise CalculatorError("exponent out of range")
        return op(left, right)
    raise CalculatorError(f"unsupported syntax {type(node).__name__}")


def evaluate(expression: str) -> float | None:
    """Evaluate a model-written arithmetic expression, or return None if it isn't one.

    Returns None rather than raising so the caller can fall back to the model's own stated
    answer. A `None` here is not an error in the run -- it means this question's answer came
    from the text channel instead of the program channel, and the eval loop records which.
    """
    if len(str(expression or "")) > _MAX_EXPR_CHARS:
        return None
    text = _clean(expression)
    if not text or len(text) > _MAX_EXPR_CHARS:
        return None
    # A bare number is a valid answer but not a *program*; callers distinguish the two, and
    # letting it through here keeps `evaluate("127.4")` from being a special case.
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    try:
        value = _walk(tree)
    except CalculatorError:
        return None
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return float(value)


def is_program(expression: str) -> bool:
    """True when the expression actually computes something rather than restating a literal.

    This gates execution, so it is not merely cosmetic: `answer_expression` is frequently just
    the answer restated ("4.35%", "-774"), and executing a restated value gains nothing while
    a mis-classified percent literal would actively corrupt it. Percent literals are therefore
    removed before the operator search, and a leading sign is skipped.
    """
    text = _PERCENT_RE.sub("", _clean(expression, expand_percent=False))
    return bool(text.strip()) and bool(re.search(r"[+\-*/]", text.strip()[1:]))
