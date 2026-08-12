"""
The evidence-bound prompt for baseline B1.

"Evidence-bound" is doing real work here, and it is the same property Contribution 2 later
tests: the generator is instructed to answer *only* from the retrieved chunks it was given,
and to say so when they don't contain the answer. Without that constraint a strong model
answers financial questions from parametric knowledge, the answer looks right, and every
downstream faithfulness measurement is meaningless because the answer never depended on the
retrieved context in the first place.

Four prompt decisions worth keeping stable, because changing them changes what B1 means:

1. **Evidence is numbered, and citations come back as those numbers.** The model cites [1],
   [3]; `generator.py` maps those back to chunk_ids. Asking it to echo raw ids like
   "table_row_7" invites transcription errors that show up as fake grounding failures.

2. **An explicit "insufficient evidence" path.** A model forced to answer always answers.
   Separating "wrong" from "correctly declined" is what lets Month 7 attribute failures to
   retrieval instead of generation -- the right-evidence-wrong-math vs no-evidence-at-all
   distinction that CLAUDE.md asks the faithfulness logs to preserve.

3. **Program-of-thought: the model returns the arithmetic, not the result** (v2). See
   `calculator.py` for the failure analysis that forced this. In short, v1 asked a 1.5B model
   to do financial arithmetic mentally and report a bare number; it copied a number from the
   evidence without computing on 24% of FinQA questions and wrote out its working -- correctly
   -- in an unparseable form on another 19%. Emitting `answer_expression` and executing it in
   Python moves the arithmetic to the one component that is exact. This is FinQA's own
   formulation (the dataset ships a gold `program` field), not a workaround.

4. **Few-shot exemplars, fixed and hand-written** (v2). At 1.5-3B, format compliance is a
   real bottleneck and instructions alone do not carry it. The exemplars are hand-written
   rather than sampled from the training split, so they leak nothing from any evaluated
   question and stay identical across datasets, models and baselines -- a constant, like the
   model choice.

Prompt text is version-tagged, and `prompt_version()` encodes the style. B1 numbers from
different prompt versions are not comparable, and the eval script records the tag alongside
the results.
"""
from __future__ import annotations

PROMPT_SCHEMA_VERSION = "v2"

# --------------------------------------------------------------------------------------
# System prompts
# --------------------------------------------------------------------------------------

_COMMON_RULES = """\
- Use only the numbered evidence. Do not use outside knowledge about the company, and do not \
infer figures that are not present in the evidence.
- Cite the evidence numbers your answer depends on. For a calculation, cite every excerpt \
whose figures you used.
- If the evidence does not contain what is needed, set insufficient_evidence to true and give \
your best partial answer. Do not guess a plausible-looking figure.
- Span questions: answer with the exact wording from the evidence, nothing more.
- Keep the answer to the value or span asked for. No explanation, no restatement of the \
question.\
"""

# v1 -- kept verbatim so the "direct" arm of the ablation is the *original* baseline rather
# than a reconstruction of it. Deleting this would make the reported improvement unverifiable.
SYSTEM_PROMPT_DIRECT = """\
You answer questions about financial reports using only the evidence provided in the user \
message. The evidence is a numbered list of excerpts (sentences and linearized table rows) \
retrieved from a single filing.

Rules:
""" + _COMMON_RULES.replace(
    "- Span questions:",
    """- Arithmetic questions: compute the value and report it as a plain number. Keep the \
units and sign convention the document uses; a value shown in a table as "(774)" is negative. \
Do not add currency symbols or thousands separators.
- Span questions:""",
)

# v2 -- program-of-thought. The behavioural instruction that matters is "do not do the
# arithmetic yourself", because a model told merely that it *may* return an expression will
# still compute mentally and return a wrong number most of the time.
SYSTEM_PROMPT_PROGRAM = """\
You answer questions about financial reports using only the evidence provided in the user \
message. The evidence is a numbered list of excerpts (sentences and linearized table rows) \
retrieved from a single filing.

Rules:
- Arithmetic questions: do NOT calculate the result yourself. Write the calculation into \
answer_expression as a plain arithmetic expression, using only numbers copied from the \
evidence, and it will be evaluated exactly. Use only + - * / and parentheses.
- A value shown in a table as "(774)" is negative: write it as -774.
- For "percentage change from A to B" write (B - A) / A * 100. For "what percentage of X is \
Y" write Y / X * 100. Percentages are reported as 42.4, not 0.424.
- Match the exact years and line items named in the question. A table row lists several years; \
take the figure under the year the question asks about, not a neighbouring one.
- Set answer_expression to "" for questions that only require reading a value or a span out \
of the evidence, and put that value or span in answer.
""" + _COMMON_RULES

# --------------------------------------------------------------------------------------
# Few-shot exemplars
# --------------------------------------------------------------------------------------
# Hand-written, deliberately generic filings. Three exemplars cover the three shapes that
# account for nearly all of FinQA and TAT-QA: a percentage change, a share-of-total ratio,
# and a read-only span. A fourth exemplar for the insufficient-evidence path was tried and
# dropped -- it raised the decline rate on answerable questions, trading real answers for
# safe ones, which is the wrong trade for a baseline meant to be beaten.
#
# **Exemplars live in the system prompt, not in user/assistant chat turns.** The chat-turn
# form is the more usual choice and it does teach the output format well -- but at 1.5B it
# caused measurable evidence contamination: an exemplar turn is structurally identical to the
# real question, so the model treated exemplar figures as retrieved evidence and answered a
# real question with `-30584 / 8920 * 100`, where 8920 appears nowhere except in the span
# exemplar below. Putting the exemplars under an explicit heading inside the system prompt
# leaves exactly one user turn containing evidence, which removes the ambiguity structurally
# rather than asking the model not to make the mistake. Format compliance was re-measured
# after the change and did not regress.

_SHOT_CHANGE = (
    """Question: what was the percentage change in research and development expense from 2015 to 2016?

Evidence:
[1] (table row) company the research and development of 2016 is $ 1245 ; the research and development of 2015 is $ 1108 ;
[2] (text) research and development expense is presented in millions of dollars .""",
    '{"answer": "12.4%", "answer_expression": "(1245 - 1108) / 1108 * 100", '
    '"evidence_numbers": [1], "insufficient_evidence": false}',
)

_SHOT_RATIO = (
    """Question: what portion of total operating expenses in 2018 was selling and administrative expense?

Evidence:
[1] (table row) company the selling and administrative of 2018 is 3402 ; the selling and administrative of 2017 is 3189 ;
[2] (table row) company the total operating expenses of 2018 is 14680 ; the total operating expenses of 2017 is 13905 ;""",
    '{"answer": "23.2%", "answer_expression": "3402 / 14680 * 100", '
    '"evidence_numbers": [1, 2], "insufficient_evidence": false}',
)

_SHOT_SPAN = (
    """Question: what was the weighted average interest rate on the notes?

Evidence:
[1] (text) the notes bear interest at a weighted average rate of 4.35% and mature in 2024 .
[2] (table row) company the long-term debt of 2018 is $ 8920 ;""",
    '{"answer": "4.35%", "answer_expression": "", "evidence_numbers": [1], '
    '"insufficient_evidence": false}',
)

# Order matters a little: the last exemplar is the one a small model imitates most strongly,
# so the span example goes last to stop it forcing an expression onto every question.
FEW_SHOT = [_SHOT_CHANGE, _SHOT_RATIO, _SHOT_SPAN]

USER_TEMPLATE = """\
Question: {question}

Evidence:
{evidence}\
"""

_FORMAT_DIRECT = (
    "\n\nRespond with a single JSON object and nothing else, in this exact form:\n"
    '{"answer": "...", "evidence_numbers": [1, 2], "insufficient_evidence": false}'
)
_FORMAT_PROGRAM = (
    "\n\nRespond with a single JSON object and nothing else, in this exact form:\n"
    '{"answer": "...", "answer_expression": "...", "evidence_numbers": [1, 2], '
    '"insufficient_evidence": false}'
)


_EXEMPLAR_HEADER = """\

Worked examples. These show the required output format only. The figures inside them belong \
to other filings and are NOT evidence -- never use a number from a worked example to answer \
the real question.
"""


def _render_exemplars(use_program: bool, n_shot: int) -> str:
    """Format the exemplars as a labelled block for the system prompt."""
    if n_shot <= 0:
        return ""
    import json as _json

    blocks = []
    for i, (user_text, assistant_text) in enumerate(FEW_SHOT[:n_shot], start=1):
        if not use_program:
            payload = _json.loads(assistant_text)
            payload.pop("answer_expression", None)
            assistant_text = _json.dumps(payload)
        blocks.append(f"Example {i}\n{user_text}\n\nOutput:\n{assistant_text}")
    return _EXEMPLAR_HEADER + "\n" + "\n\n".join(blocks) + "\n"


def prompt_version(use_program: bool = True, n_shot: int = 3) -> str:
    """Version tag recorded with every generated answer.

    Encoding the style in the tag is what makes the ablation auditable after the fact: a
    results file alone tells you which arm produced it, without cross-referencing a run log.
    """
    style = "pot" if use_program else "direct"
    if not use_program and n_shot == 0:
        return "b1-evidence-bound-v1"      # the original baseline, byte-identical prompt
    return f"b1-{style}-{n_shot}shot-{PROMPT_SCHEMA_VERSION}"


# Default style, and the tag imported by anything that doesn't care about the ablation.
PROMPT_VERSION = prompt_version()
SYSTEM_PROMPT = SYSTEM_PROMPT_PROGRAM


def system_prompt(use_program: bool = True, n_shot: int = 0) -> str:
    base = SYSTEM_PROMPT_PROGRAM if use_program else SYSTEM_PROMPT_DIRECT
    return base + _render_exemplars(use_program, n_shot)


def format_evidence(retrieved) -> str:
    """Number the retrieved chunks 1..k for citation. Order is retrieval rank, so [1] is the
    top-scored chunk -- which also makes the transcript readable when debugging a bad answer."""
    lines = []
    for i, hit in enumerate(retrieved, start=1):
        kind = "table row" if hit.chunk.chunk_type == "table_row" else "text"
        lines.append(f"[{i}] ({kind}) {hit.chunk.text}")
    return "\n".join(lines) if lines else "(no evidence retrieved)"


def build_user_message(question: str, retrieved, use_program: bool = True) -> str:
    """The question, its numbered evidence, and the required response shape."""
    body = USER_TEMPLATE.format(question=question, evidence=format_evidence(retrieved))
    return body + (_FORMAT_PROGRAM if use_program else _FORMAT_DIRECT)


def build_messages(question: str, retrieved, use_program: bool = True,
                   n_shot: int = 3) -> list[dict]:
    """Full chat message list: one system turn (rules + worked examples) and one user turn.

    Exactly one turn carries evidence, by design -- see the note above FEW_SHOT for the
    contamination this prevents.
    """
    return [
        {"role": "system", "content": system_prompt(use_program, max(0, n_shot))},
        {"role": "user", "content": build_user_message(question, retrieved, use_program)},
    ]


# Structured output schema. Constraining the response shape means the eval loop never has to
# regex an answer out of prose -- and `evidence_numbers` gives the grounding layer (Month 5) a
# model-declared citation set to check sentence-level entailment against.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer value or span. Plain number for arithmetic questions.",
        },
        "answer_expression": {
            "type": "string",
            "description": (
                "Arithmetic expression over numbers copied from the evidence, evaluated "
                "exactly by the caller. Empty string for read-only questions."
            ),
        },
        "evidence_numbers": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Numbers of the evidence excerpts the answer depends on.",
        },
        "insufficient_evidence": {
            "type": "boolean",
            "description": "True when the provided evidence does not contain the answer.",
        },
    },
    "required": ["answer", "answer_expression", "evidence_numbers", "insufficient_evidence"],
    "additionalProperties": False,
}
