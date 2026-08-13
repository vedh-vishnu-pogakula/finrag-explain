"""
RAGAS wired to a local judge, with a hard guard against ever reaching a paid API.

**This module is where the project's zero-budget constraint is most likely to fail, and it is
designed around that.** RAGAS's own factory signature is
`llm_factory(model, provider="openai", ...)` -- the default provider is a billed one, in code,
not just in documentation. Every published RAGAS example calls `evaluate(dataset, metrics)`
with no `llm=`, and that path constructs an OpenAI judge and charges on the first call. A
student following any tutorial verbatim would be billed before seeing a single score.

So nothing here is left to a default:

1. **Every metric is constructed with an explicit local `llm=`.** `verify_metrics_are_local()`
   raises if a metric reaches evaluation without one, so a forgotten assignment fails loudly
   instead of silently falling back.
2. **Credentials are neutralised for the duration of the run** by `no_paid_providers()`. Any
   code path that still tries to reach a paid API gets an obviously-invalid key and fails with
   an auth error rather than succeeding and billing. Belt and braces: guard (1) is the design,
   guard (2) is what catches a RAGAS internal we did not anticipate.
3. **Telemetry is switched off.** RAGAS phones home by default; that is not a cost, but it is
   not something a reproducible evaluation should do silently either.

**Which judge, and why it is split in two.** Faithfulness is a two-stage metric: an LLM
decomposes the answer into atomic statements, then something verifies each statement against
the retrieved context. Those two stages have very different requirements:

  * *Decomposition* is mechanical -- restate an answer as a list of claims. A local
    Qwen2.5-7B-Instruct is adequate, and using the same model that generated the answer is
    acceptable here because the step involves no judgement about correctness.
  * *Verification* is the actual measurement, and there the generator must not grade itself.
    `FaithfulnesswithHHEM` replaces the LLM verifier with Vectara's hallucination-evaluation
    model -- a purpose-built NLI classifier from an unrelated lineage. That independence is
    what makes the Contribution 2 experiment meaningful rather than circular.

Note that HHEM loads with `trust_remote_code=True` (RAGAS's own call, not ours), so it
executes code from the model repository. That is Vectara's published model and standard
practice for it, but it is a real supply-chain consideration and belongs in the paper's
reproducibility notes rather than being discovered later.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / "generation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Every credential RAGAS or its langchain dependencies could pick up out of the environment.
# Removing them is not paranoia: langchain clients read these implicitly, so a key exported
# for an unrelated project is enough to turn an accidental fallback into a real charge.
_PAID_CREDENTIAL_VARS = (
    "OPENAI_API_KEY", "OPENAI_ORGANIZATION", "OPENAI_BASE_URL",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
    "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
    "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY",
)

# Deliberately invalid. If some RAGAS internal still reaches for OpenAI, it gets a 401 --
# loud, free and traceable -- instead of a working key and an invoice.
_SENTINEL_KEY = "sk-LOCAL-JUDGE-ONLY-THIS-PROJECT-MUST-NOT-BILL"


class PaidProviderError(RuntimeError):
    """Raised when an evaluation is about to use a metric without a local judge."""


@contextmanager
def no_paid_providers():
    """Neutralise paid-API credentials and telemetry for the duration of a RAGAS run."""
    saved = {name: os.environ.pop(name, None) for name in _PAID_CREDENTIAL_VARS}
    saved_track = os.environ.get("RAGAS_DO_NOT_TRACK")
    os.environ["OPENAI_API_KEY"] = _SENTINEL_KEY
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    try:
        yield
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
        if saved_track is None:
            os.environ.pop("RAGAS_DO_NOT_TRACK", None)
        else:
            os.environ["RAGAS_DO_NOT_TRACK"] = saved_track


def _base_ragas_llm():
    """Imported lazily so this module can be inspected without pulling in RAGAS."""
    from ragas.llms.base import BaseRagasLLM

    return BaseRagasLLM


class LocalRagasLLM(_base_ragas_llm()):
    """A RAGAS-compatible judge backed by the local open-weights model.

    **Subclasses `BaseRagasLLM` rather than duck-typing it.** Implementing only the three
    abstract methods looks sufficient -- and passes an `__abstractmethods__` check -- but
    RAGAS's prompts call `llm.generate()`, a *concrete* method on the base that wraps
    `agenerate_text` with retry, temperature defaulting and a finish check. A duck-typed class
    has no `generate`, and the failure surfaces one call deep as
    `AttributeError: 'LocalRagasLLM' object has no attribute 'generate'`.

    The judge runs on the same weights already loaded for generation -- no second download, no
    second model in VRAM, no network.

    `temperature` is accepted and ignored: decoding here is greedy for the same reason B1's is
    (CLAUDE.md). A judge that scores differently on every run cannot support a claim about
    whether a *metric* moved when evidence was perturbed -- the noise would be
    indistinguishable from the effect Month 6 is trying to measure.
    """

    # RAGAS's statement decomposition emits a JSON list and needs more room than a B1 answer.
    DEFAULT_JUDGE_TOKENS = 1024

    def __init__(self, generator=None, max_new_tokens: int = DEFAULT_JUDGE_TOKENS):
        from generator import LocalGenerator

        super().__init__()          # sets run_config / cache defaults on the base dataclass
        self.generator = generator if generator is not None else LocalGenerator.from_config()
        self.max_new_tokens = max_new_tokens

    # -- BaseRagasLLM interface ------------------------------------------------------------
    def generate_text(self, prompt, n: int = 1, temperature: float = 0.01,
                      stop=None, callbacks=None):
        from langchain_core.outputs import Generation, LLMResult

        text = self.generator.complete(_prompt_to_text(prompt), self.max_new_tokens)
        # n>1 would need sampling, which is disabled on purpose; RAGAS only asks for n>1 on
        # metrics this project does not use, so returning the greedy completion n times is
        # honest about what the model actually produced.
        return LLMResult(generations=[[Generation(text=text) for _ in range(max(1, n))]])

    async def agenerate_text(self, prompt, n: int = 1, temperature=0.01,
                             stop=None, callbacks=None):
        # transformers generation is synchronous and GPU-bound; RAGAS's executor awaits this,
        # so a plain synchronous call inside the coroutine is correct. Wrapping it in a thread
        # pool would let several questions contend for one GPU and slow the run down.
        return self.generate_text(prompt, n=n, temperature=temperature, stop=stop,
                                  callbacks=callbacks)

    def is_finished(self, response) -> bool:
        """Local generation always returns a complete response or raises, so there is no
        provider-side truncation signal to inspect."""
        return True

    @property
    def name(self) -> str:
        return f"local:{getattr(self.generator, 'model', 'unknown')}"


def _prompt_to_text(prompt) -> str:
    """RAGAS passes a langchain PromptValue; older paths pass a bare string."""
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    return str(prompt)


def build_faithfulness_metric(llm, use_hhem: bool = True, device: str = "cpu"):
    """The metric Contribution 2 puts under test.

    `use_hhem=True` keeps statement verification on Vectara's NLI model rather than on the
    judge LLM, which is what stops the generator from grading its own answers. Set it to False
    only to *compare* the two verifiers -- that contrast is itself a reportable result, not a
    fallback.
    """
    from ragas.metrics._faithfulness import Faithfulness, FaithfulnesswithHHEM

    if use_hhem:
        return FaithfulnesswithHHEM(llm=llm, device=device)
    return Faithfulness(llm=llm)


def verify_metrics_are_local(metrics) -> None:
    """Refuse to start an evaluation that could reach a billed provider.

    Checked before `evaluate()` rather than after, because "after" means the charge already
    happened. A metric with `llm=None` is the specific failure mode: RAGAS fills it in from
    its own default, and its own default is OpenAI.
    """
    for metric in metrics:
        llm = getattr(metric, "llm", "absent")
        if llm == "absent":
            continue                      # a metric that needs no LLM cannot bill
        if llm is None:
            raise PaidProviderError(
                f"metric {getattr(metric, 'name', metric)!r} has llm=None. RAGAS would fill "
                "that in with its OpenAI default and bill on the first call. Construct it "
                "with an explicit local judge -- see build_faithfulness_metric()."
            )
        module = type(llm).__module__ or ""
        if not isinstance(llm, LocalRagasLLM) and "ragas" not in module:
            # Not fatal -- a free hosted judge is an allowed fallback per CLAUDE.md -- but it
            # must be a deliberate choice, so say what is about to happen.
            print(f"[ragas] NOTE: metric {getattr(metric, 'name', metric)!r} uses judge "
                  f"{type(llm).__name__} from {module}. Confirm it is free before running.")


def default_run_config(max_workers: int = 2, timeout: int = 180, max_retries: int = 5):
    """Throttled RunConfig.

    RAGAS defaults to 16 concurrent workers, which is tuned for a hosted API. Against one
    local GPU that is 16 requests queueing on the same device -- no faster, and it turns a
    single slow question into a timeout cascade that retries and slows things further. Two
    workers is the CLAUDE.md guardrail.
    """
    from ragas.run_config import RunConfig

    return RunConfig(max_workers=max_workers, timeout=timeout, max_retries=max_retries)
