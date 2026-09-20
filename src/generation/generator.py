"""
Generation layer for baseline B1: retrieved chunks in, evidence-bound answer out.

**This project has a hard zero-budget constraint, so the default generator is a local
open-weights model run through `transformers` -- no paid API, no metered tokens, no billing
account anywhere in the pipeline.** Three implementations share one interface:

  LocalGenerator  -- default. An instruction-tuned open-weights model on your own hardware
                     (Apple MPS / Colab's free T4 / CPU). Free forever, and reproducible,
                     which is the stronger claim to defend in a viva: anyone can re-run it.
  ApiGenerator    -- opt-in only (`generation.provider: api`). Kept because it is the same
                     code shape and useful as a quality ceiling to *cite* -- but it bills a
                     real account, so nothing in the default config path can reach it.
  StubGenerator   -- offline echo of the top chunk. Not a baseline; it exists so tests and a
                     model-less demo can exercise the whole retrieve -> generate -> score path
                     in under a second.

Model choice is pinned in configs/config.yaml, and changing it invalidates cross-baseline
comparability -- it's an experimental constant, not a tuning knob.

What this module deliberately does *not* do:

* It is never called inside a perturbation loop. Month 4 attribution perturbs the *query* and
  re-scores against cached chunk embeddings (`Retriever.score_query_variants`); Month 6
  perturbs the *evidence* and re-runs RAGAS. Calling a generator per perturbation is the
  compute-exhausting anti-pattern the guardrails single out. If you find yourself importing a
  generator into `src/attribution/`, stop.
* It does not retry on a wrong answer or self-critique. B1 is the plain comparison floor.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / "generation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from calculator import evaluate as eval_expression, is_program  # noqa: E402
from config_utils import get as cfg_get, load_config  # noqa: E402
from prompts import (  # noqa: E402
    ANSWER_SCHEMA,
    PROMPT_VERSION,
    build_messages,
    prompt_version,
)

# Local default. Qwen2.5-1.5B-Instruct is chosen over a same-size Flan-T5 or base LLaMA for
# three reasons that matter here: it follows an instruction format reliably enough to emit JSON
# at 1.5B, it is genuinely decent at the small multi-step arithmetic FinQA needs, and it fits
# in memory on a laptop (~3GB in fp16) as well as on Colab's free T4. Step up to
# Qwen2.5-3B-Instruct or 7B on a GPU if quality is the bottleneck -- the interface is identical
# and only configs/config.yaml changes.
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Answers are a value plus a short citation list. 256 new tokens is generous for that and keeps
# generation fast; it is a cap on the *answer*, not on reasoning, because this model has no
# separate thinking channel.
DEFAULT_MAX_NEW_TOKENS = 256

DEFAULT_API_MODEL = "claude-opus-5"
DEFAULT_API_MAX_TOKENS = 8000
# Claude Opus 5's safety classifiers can decline a request (HTTP 200, stop_reason "refusal").
# Only reachable on the opt-in API path.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class GeneratedAnswer:
    """One generated answer plus the trail needed to grade and audit it.

    `answer` is what gets scored. Under program-of-thought it is the *executed* value when the
    model supplied a usable expression, and the model's own stated answer otherwise -- with
    `answer_source` recording which, and `answer_raw` preserving what the model said before
    the calculator overrode it. Keeping all three is what makes the improvement auditable:
    the program-channel share and the raw-vs-executed disagreement rate are both reportable,
    and neither is recoverable if the executed value simply overwrites the field.
    """
    answer: str
    cited_chunk_ids: list = field(default_factory=list)
    insufficient_evidence: bool = False
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    error: str | None = None
    answer_expression: str = ""
    answer_raw: str = ""
    answer_source: str = "text"        # "program" | "text"

    def to_json(self) -> dict:
        return {
            "answer": self.answer,
            "cited_chunk_ids": self.cited_chunk_ids,
            "insufficient_evidence": self.insufficient_evidence,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "error": self.error,
            "answer_expression": self.answer_expression,
            "answer_raw": self.answer_raw,
            "answer_source": self.answer_source,
        }


def _format_value(value: float) -> str:
    """Render an executed value as a plain string for scoring.

    Rounded to four decimals to avoid handing the metric a float-repr artifact like
    12.400000000000002, which is well inside the metric's 1% tolerance but reads as noise in
    the per-question logs a human has to audit.
    """
    rounded = round(value, 4)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


def _apply_program(payload: dict) -> tuple[str, str, str, str]:
    """Resolve the answer from the program channel when one is available.

    Returns (answer, raw_answer, expression, source). The expression wins over the stated
    answer whenever it evaluates *and* actually computes something -- a bare literal in
    `answer_expression` is not a program and carries no more information than `answer` does,
    so it is not allowed to overwrite it.
    """
    raw = str(payload.get("answer", "")).strip()
    expression = str(payload.get("answer_expression", "") or "").strip()
    if expression and is_program(expression):
        value = eval_expression(expression)
        if value is not None:
            return _format_value(value), raw, expression, "program"
    return raw, raw, expression, "text"


def build_generator(cfg: dict | None = None, provider: str | None = None, **overrides):
    """Factory the eval scripts and demo call. Defaults to the free local model.

    The API provider is reachable only by explicitly setting `generation.provider: api` (or
    passing --generator api), and it announces the cost when it starts. That asymmetry is on
    purpose: under a zero-budget constraint, spending money must never be something that
    happens because a default was left alone.

    `overrides` (model / use_program / n_shot) exist for the prompt ablation, which has to vary
    those without editing the config the frozen baseline reads from. Values of None are dropped
    so a caller can pass unset CLI flags straight through.
    """
    cfg = cfg or load_config()
    provider = provider or cfg_get(cfg, "generation.provider", "local")
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if provider == "local":
        return LocalGenerator.from_config(cfg, **overrides)
    if provider == "api":
        print("[generator] WARNING: provider='api' bills a real account. The project's "
              "zero-cost default is provider='local'.")
        return ApiGenerator.from_config(cfg, **{k: v for k, v in overrides.items()
                                                if k in {"model", "max_tokens"}})
    if provider == "stub":
        return StubGenerator()
    raise ValueError(f"unknown generation provider {provider!r} (local | api | stub)")


# JSON only permits \" \\ \/ \b \f \n \r \t \uXXXX. Small models routinely emit LaTeX-style
# escapes such as "\%" and "\$" inside financial answers, which makes the whole object
# unparseable even though its content is perfectly good.
_BAD_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _extract_json(text: str) -> dict | None:
    """Pull the first balanced {...} object out of a model's raw output.

    A hosted API can constrain decoding to a schema; a 1.5B local model cannot, so it will
    sometimes wrap the JSON in prose or a ```json fence. Brace-matching (rather than a regex)
    is what makes this survive nested objects and stray braces inside string values.

    On a decode failure the candidate is retried once with invalid escape sequences stripped.
    That is a deliberately narrow repair -- it recovers `{"answer": "$ 11.6 \\% "}`, which is
    a formatting slip rather than a wrong answer, without inventing content the model didn't
    produce.
    """
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    for attempt in (candidate, _BAD_ESCAPE_RE.sub("", candidate)):
                        try:
                            parsed = json.loads(attempt)
                            return parsed if isinstance(parsed, dict) else None
                        except json.JSONDecodeError:
                            continue
                    break
        start = text.find("{", start + 1)
    return None


# End-of-turn markers used by common instruct chat templates. Only the ones that exist in a
# given tokenizer's vocabulary are used, so this is safe for any model.
_END_OF_TURN_TOKENS = ("<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>", "</s>",
                       "<end_of_turn>")


def end_of_turn_ids(tokenizer) -> list[int]:
    """EOS plus every end-of-turn token this tokenizer actually has, de-duplicated."""
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in _END_OF_TURN_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or tid == unk or tid < 0 or tid in ids:
            continue
        ids.append(int(tid))
    return ids


class LocalGenerator:
    """Evidence-bound generation on a local open-weights model. Free, offline, reproducible.

    Decoding is greedy (`do_sample=False`): a research baseline that returns a different answer
    on every run can't be compared against B2/B3, and sampling would make the Month 6
    perturbation results indistinguishable from decoding noise.
    """

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, device: str | None = None,
                 use_program: bool = True, n_shot: int = 3, load_in_4bit: bool = False):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.use_program = use_program
        self.n_shot = n_shot
        self.load_in_4bit = load_in_4bit
        self._pipe = None
        self._eos_ids: list[int] | None = None

    @property
    def prompt_version(self) -> str:
        return prompt_version(self.use_program, self.n_shot)

    @classmethod
    def from_config(cls, cfg: dict | None = None, **overrides):
        cfg = cfg or load_config()
        kwargs = dict(
            model=cfg_get(cfg, "generation.local_model", DEFAULT_LOCAL_MODEL),
            max_new_tokens=int(cfg_get(cfg, "generation.max_new_tokens",
                                       DEFAULT_MAX_NEW_TOKENS)),
            device=cfg_get(cfg, "generation.device"),
            use_program=bool(cfg_get(cfg, "generation.use_program", True)),
            n_shot=int(cfg_get(cfg, "generation.n_shot", 3)),
            load_in_4bit=bool(cfg_get(cfg, "generation.load_in_4bit", False)),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"      # Colab's free T4
        if torch.backends.mps.is_available():
            return "mps"       # Apple Silicon
        return "cpu"

    @property
    def pipe(self):
        """Loaded once and reused. Reloading per question would dominate the runtime and is
        the same mistake the demo avoids with st.cache_resource."""
        if self._pipe is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self._resolve_device()
            tokenizer = AutoTokenizer.from_pretrained(self.model)

            if self.load_in_4bit:
                # 4-bit NF4 is what makes a 7B model fit in a free Colab T4's 15GB alongside
                # its KV cache -- fp16 weights alone are ~15GB and OOM before the first token.
                # Quantization is deterministic, so greedy decoding stays reproducible; but it
                # is part of the frozen generation config, exactly like the model name, and
                # B1/B2/B3 must all be produced under the same setting.
                if device != "cuda":
                    raise RuntimeError(
                        f"generation.load_in_4bit requires a CUDA GPU (bitsandbytes), but the "
                        f"resolved device is {device!r}. Run this on Colab's free T4, or set "
                        f"load_in_4bit: false and use a smaller model locally."
                    )
                from transformers import BitsAndBytesConfig

                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    # T4 is Turing: fp16 compute, no bf16.
                    bnb_4bit_compute_dtype=torch.float16,
                )
                # device_map places the quantized shards; calling .to(device) afterwards is an
                # error on a bitsandbytes model, so this branch must not fall through.
                model = AutoModelForCausalLM.from_pretrained(
                    self.model, quantization_config=quant, device_map={"": 0},
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    dtype=torch.float16 if device != "cpu" else torch.float32,
                ).to(device)

            model.eval()
            self._pipe = (tokenizer, model, device)
            self._eos_ids = end_of_turn_ids(tokenizer)
        return self._pipe

    def generate(self, question: str, retrieved) -> GeneratedAnswer:
        messages = build_messages(question, retrieved,
                                  use_program=self.use_program, n_shot=self.n_shot)
        try:
            raw = self._run(messages)
        except Exception as exc:
            return GeneratedAnswer(answer="", model=self.model,
                                   prompt_version=self.prompt_version,
                                   error=f"{type(exc).__name__}: {exc}")

        payload = _extract_json(raw)
        if payload is None:
            # A small model that ignored the format instruction still usually states the answer
            # in prose -- and under program-of-thought that prose is often a bare arithmetic
            # expression, which the calculator can still execute. Salvaging it here costs
            # nothing and is recorded as `unparseable_json` either way, so the format-compliance
            # rate stays an honest finding about the model rather than being papered over.
            text = raw.strip()[:200]
            salvaged = eval_expression(text) if self.use_program and is_program(text) else None
            return GeneratedAnswer(
                answer=_format_value(salvaged) if salvaged is not None else text,
                answer_raw=text,
                answer_expression=text if salvaged is not None else "",
                answer_source="program" if salvaged is not None else "text",
                model=self.model, prompt_version=self.prompt_version,
                error="unparseable_json",
            )

        answer, answer_raw, expression, source = _apply_program(payload)
        return GeneratedAnswer(
            answer=answer,
            answer_raw=answer_raw,
            answer_expression=expression,
            answer_source=source,
            cited_chunk_ids=_resolve_citations(payload.get("evidence_numbers", []), retrieved),
            insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
            model=self.model,
            prompt_version=self.prompt_version,
            stop_reason="stop",
        )

    def complete(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Raw text completion, no B1 answer schema attached.

        Month 5 needs this: RAGAS's faithfulness metric decomposes an answer into statements
        with its own prompts, and it needs a plain text-in/text-out model rather than the
        evidence-bound JSON path `generate()` provides. Routing it through the same loaded
        model is what keeps the judge free -- a second model would double the download and
        the VRAM for no benefit.
        """
        previous, self.max_new_tokens = self.max_new_tokens, max_new_tokens or self.max_new_tokens
        try:
            return self._run([{"role": "user", "content": prompt}])
        finally:
            self.max_new_tokens = previous

    def _run(self, messages: list[dict]) -> str:
        import torch

        tokenizer, model, device = self.pipe
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,             # deterministic -- see class docstring
                pad_token_id=tokenizer.eos_token_id,
                # Stop on the chat template's end-of-turn token as well as EOS. For Qwen they
                # are the same token, so B1/B2 outputs are unchanged; for a judge whose template
                # ends turns with a different token (Falcon3, Llama-3), stopping only on EOS
                # runs every call to max_new_tokens -- a 1024-token verdict 500 times over is
                # what turned a 1.5 h Colab cell into 5 h with nothing written.
                eos_token_id=self._eos_ids,
            )
        # Decode only the newly generated tokens, not the echoed prompt.
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)


class ApiGenerator:
    """Opt-in hosted-API generation. **Costs money** -- unreachable from the default config."""

    def __init__(self, model: str = DEFAULT_API_MODEL, max_tokens: int = DEFAULT_API_MAX_TOKENS,
                 refusal_fallback: bool = True, client=None):
        self.model = model
        self.max_tokens = max_tokens
        self.refusal_fallback = refusal_fallback
        self._client = client

    @classmethod
    def from_config(cls, cfg: dict | None = None, **overrides):
        cfg = cfg or load_config()
        kwargs = dict(
            model=cfg_get(cfg, "generation.api_model", DEFAULT_API_MODEL),
            max_tokens=int(cfg_get(cfg, "generation.max_tokens", DEFAULT_API_MAX_TOKENS)),
            refusal_fallback=bool(cfg_get(cfg, "generation.refusal_fallback", True)),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def client(self):
        if self._client is None:
            import anthropic

            # Credentials resolve from the environment (ANTHROPIC_API_KEY, or an `ant auth
            # login` profile) -- never hardcode a key in the repo.
            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, question: str, retrieved) -> GeneratedAnswer:
        """Answer `question` from `retrieved` (a rank-ordered list of RetrievedChunk).

        Returns a GeneratedAnswer even on failure, with `error` set -- the eval loop checkpoints
        per question and must not lose 200 completed questions to one bad API call.
        """
        messages = build_messages(question, retrieved, use_program=True, n_shot=3)
        request = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=messages[0]["content"],
            messages=messages[1:],
            output_config={"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        )

        try:
            response = self._call(request)
        except Exception as exc:  # surfaced per question, not fatal to the run
            return GeneratedAnswer(answer="", model=self.model, error=f"{type(exc).__name__}: {exc}")

        return self._parse(response, retrieved)

    def _call(self, request: dict):
        """One API call, with the refusal fallback when it is available.

        The beta is disabled for the rest of the run if the account rejects it, rather than
        failing every question after the first -- the fallback is insurance, not a requirement.
        """
        if self.refusal_fallback:
            try:
                return self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **request
                )
            except Exception as exc:
                if "fallback" not in str(exc).lower() and "beta" not in str(exc).lower():
                    raise
                print(f"[generator] refusal fallback unavailable ({exc}); continuing without it")
                self.refusal_fallback = False
        return self.client.messages.create(**request)

    def _parse(self, response, retrieved) -> GeneratedAnswer:
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }
        stop_reason = getattr(response, "stop_reason", None)

        # Check stop_reason before touching content: a refusal returns HTTP 200 with empty or
        # partial content, so indexing content[0] blindly is how this breaks in production.
        if stop_reason == "refusal":
            return GeneratedAnswer(
                answer="", model=getattr(response, "model", self.model),
                stop_reason=stop_reason, usage=usage, error="refusal",
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Shouldn't happen under a json_schema output_config, but a truncated response
            # (stop_reason "max_tokens") produces invalid JSON -- record it rather than crash.
            return GeneratedAnswer(
                answer=text.strip(), model=getattr(response, "model", self.model),
                stop_reason=stop_reason, usage=usage, error="unparseable_json",
            )

        answer, answer_raw, expression, source = _apply_program(payload)
        return GeneratedAnswer(
            answer=answer,
            answer_raw=answer_raw,
            answer_expression=expression,
            answer_source=source,
            cited_chunk_ids=_resolve_citations(payload.get("evidence_numbers", []), retrieved),
            insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
            model=getattr(response, "model", self.model),
            stop_reason=stop_reason,
            usage=usage,
        )


def _resolve_citations(numbers, retrieved) -> list:
    """Map the 1-based evidence numbers the model cites back to chunk_ids. Out-of-range numbers
    are dropped rather than repaired -- a hallucinated citation index is itself a signal, and
    silently snapping it to a real chunk would manufacture grounding that isn't there."""
    ids = []
    for n in numbers:
        if isinstance(n, int) and 1 <= n <= len(retrieved):
            ids.append(retrieved[n - 1].chunk.chunk_id)
    return ids


class StubGenerator:
    """Offline stand-in: echoes the top retrieved chunk as the answer.

    Same interface as Generator so the tests and a keyless demo can run the whole
    retrieve -> generate -> score path. It is not a baseline and its answer scores are
    meaningless -- `eval/baselines/run_b1_rag.py` suffixes its output files accordingly.
    """

    model = "stub-generator-offline"

    def __init__(self, max_tokens: int = 0):
        self.max_tokens = max_tokens

    def generate(self, question: str, retrieved) -> GeneratedAnswer:
        if not retrieved:
            return GeneratedAnswer(answer="", model=self.model, insufficient_evidence=True)
        top = retrieved[0]
        return GeneratedAnswer(
            answer=top.chunk.text[:200],
            cited_chunk_ids=[top.chunk.chunk_id],
            model=self.model,
        )
