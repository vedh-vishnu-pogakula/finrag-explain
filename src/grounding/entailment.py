"""
Local natural-language-inference scoring for the evidence-grounding layer.

Grounding needs one primitive: given an evidence sentence (premise) and an answer claim
(hypothesis), how strongly does the evidence *entail* the claim. That is textual entailment,
and a cross-encoder NLI model does it better, faster and more cheaply than an LLM prompted to
say "yes/no" -- it is trained for exactly this, it returns a calibrated probability rather
than a token, and it runs on a laptop for free.

**Why not reuse the generator, or RAGAS's judge.** Two independence constraints, and both are
methodological rather than technical:

1. The generator must not grade its own answers. A model asked whether its own output follows
   from the evidence shares every blind spot that produced the output.
2. This scorer must stay independent of whatever RAGAS uses internally. Contribution 2
   perturbs the evidence *this layer* says matters and checks whether *RAGAS's* score moves in
   response. If both sides ran the same model, that experiment would be testing a model
   against itself and could not fail -- so `configs/config.yaml` deliberately points grounding
   at a DeBERTa-MNLI cross-encoder while RAGAS's faithfulness verification uses Vectara HHEM.

**The label-order trap.** NLI checkpoints do not agree on which output index means
"entailment" -- `microsoft/deberta-large-mnli` is (contradiction, neutral, entailment) while
several community checkpoints are (contradiction, entailment, neutral). Hard-coding index 2,
which is what most example code does, silently returns the *neutral* probability for half the
models on the Hub and every downstream number becomes quiet nonsense. The index is therefore
resolved from `model.config.id2label` at load time, and a checkpoint whose labels cannot be
interpreted raises rather than guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# DeBERTa-v3-base fine-tuned on MNLI+FEVER+ANLI. Chosen over a large checkpoint because
# grounding runs over every (claim, evidence sentence) pair for 250 questions and the base
# model is several times faster for a small accuracy cost; over HHEM because RAGAS uses HHEM
# and the two must stay independent (see module docstring).
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512

_ENTAIL_RE = re.compile(r"entail", re.I)
_CONTRA_RE = re.compile(r"contradict|refut", re.I)


@dataclass
class EntailmentResult:
    """Probabilities for one (premise, hypothesis) pair."""
    entailment: float
    contradiction: float
    neutral: float

    def to_json(self) -> dict:
        return {"entailment": round(self.entailment, 4),
                "contradiction": round(self.contradiction, 4),
                "neutral": round(self.neutral, 4)}


def _resolve_label_indices(id2label: dict) -> tuple[int, int, int]:
    """Map (entailment, contradiction, neutral) onto this checkpoint's output indices.

    Raises rather than defaulting: a wrong index here does not fail loudly, it returns a
    plausible-looking probability for the wrong class, and every grounding number downstream
    becomes wrong in a way no test would catch.
    """
    entail = contra = neutral = None
    for idx, label in id2label.items():
        idx = int(idx)
        if _ENTAIL_RE.search(str(label)):
            entail = idx
        elif _CONTRA_RE.search(str(label)):
            contra = idx
        else:
            neutral = idx
    if entail is None or contra is None or neutral is None:
        raise ValueError(
            f"cannot identify NLI labels from id2label={id2label!r}. Grounding needs to know "
            "which output index is entailment; guessing would silently score the wrong class."
        )
    return entail, contra, neutral


class EntailmentScorer:
    """Batched local NLI scoring. Free, offline after the first download, deterministic."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 max_length: int = DEFAULT_MAX_LENGTH, device: str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._loaded = None

    @classmethod
    def from_config(cls, cfg: dict | None = None, **overrides):
        from config_utils import get as cfg_get, load_config

        cfg = cfg or load_config()
        kwargs = dict(
            model_name=cfg_get(cfg, "grounding.nli_model", DEFAULT_NLI_MODEL),
            batch_size=int(cfg_get(cfg, "grounding.batch_size", DEFAULT_BATCH_SIZE)),
            max_length=int(cfg_get(cfg, "grounding.max_length", DEFAULT_MAX_LENGTH)),
            device=cfg_get(cfg, "grounding.device"),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def loaded(self):
        if self._loaded is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = self._resolve_device()
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.to(device)
            model.eval()
            indices = _resolve_label_indices(model.config.id2label)
            self._loaded = (tokenizer, model, device, indices, torch)
        return self._loaded

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[EntailmentResult]:
        """Score (premise, hypothesis) pairs in batches.

        Batching is not an optimisation detail here -- grounding a 250-question run means
        tens of thousands of pairs, and one forward pass per pair would dominate the whole
        Month 5 runtime. Same principle as the attribution engine: collect the work, then do
        it in as few passes as possible.
        """
        if not pairs:
            return []
        tokenizer, model, device, (ent, con, neu), torch = self.loaded
        results: list[EntailmentResult] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            encoded = tokenizer(
                [p for p, _ in batch], [h for _, h in batch],
                return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length,
            ).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            for row in probs:
                results.append(EntailmentResult(entailment=float(row[ent]),
                                                contradiction=float(row[con]),
                                                neutral=float(row[neu])))
        return results


class StubEntailmentScorer:
    """Offline stand-in scoring lexical overlap instead of entailment.

    It exists so the grounding pipeline, its aggregation and the B2 runner can be tested
    end to end in under a second with no model download -- the same role `HashEmbedder` plays
    for retrieval. Its scores are not entailment and must never appear in reported results;
    the eval scripts suffix their output files when it is used.
    """

    model_name = "stub-entailment-offline"

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE, **_):
        self.batch_size = batch_size

    @staticmethod
    def _tokens(text: str) -> set:
        return {t for t in re.findall(r"[\w.%-]+", str(text).lower()) if len(t) > 1}

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[EntailmentResult]:
        results = []
        for premise, hypothesis in pairs:
            hyp = self._tokens(hypothesis)
            overlap = len(hyp & self._tokens(premise)) / len(hyp) if hyp else 0.0
            results.append(EntailmentResult(entailment=overlap,
                                            contradiction=0.0,
                                            neutral=1.0 - overlap))
        return results


def build_scorer(cfg: dict | None = None, backend: str | None = None):
    """Factory. Defaults to the real local NLI model; `stub` is opt-in and offline."""
    from config_utils import get as cfg_get, load_config

    cfg = cfg or load_config()
    backend = backend or cfg_get(cfg, "grounding.backend", "nli")
    if backend == "nli":
        return EntailmentScorer.from_config(cfg)
    if backend == "stub":
        return StubEntailmentScorer()
    raise ValueError(f"unknown grounding backend {backend!r} (nli | stub)")
