"""
Embedding layer for the retriever (Month 3, baseline B1).

Two things in here are load-bearing for later months, not just conveniences:

1. **Token-limit enforcement uses the real tokenizer.** Per the CLAUDE.md guardrail, chunk
   length is capped by the embedding model's own tokenizer, never by word or character
   counts. A linearized FinQA table row like "company the net revenue of 2008 is 1,234 ;
   the net revenue of 2009 is ..." tokenizes far denser than prose (digits and commas each
   cost a token), so a char-based cap would silently truncate exactly the numeric cells
   retrieval depends on. `truncation_report()` tells you how often the cap actually bites.

2. **Everything comes back L2-normalized float32.** That makes inner product == cosine
   similarity, so a plain matrix multiply against cached chunk embeddings *is* the
   retriever's scoring function. Month 4's attribution loop depends on that: it batch-embeds
   query perturbations once and re-scores with one matmul, instead of re-running retrieval
   (let alone the generator) per perturbation.

`HashEmbedder` at the bottom is a deterministic, offline stand-in with the same interface --
used by the tests so they stay fast and network-free like the loader tests, and usable as a
fallback in the demo when no model has been downloaded. It is NOT a serious retriever; never
report numbers from it.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models are trained asymmetrically: passages are embedded bare, but queries are meant to
# carry this instruction prefix. Skipping it costs a few points of recall for free. Not applied
# to chunks -- only to queries. all-MiniLM-L6-v2 is symmetric and wants no prefix at all, hence
# the per-model lookup rather than a hardcoded constant.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
}


class Embedder:
    """Wraps a sentence-transformers model. Loads lazily so constructing an Embedder (e.g. to
    read `.model_name` in a config check) doesn't pull ~130MB off the Hub."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = 256,
        batch_size: int = 64,
        device: str | None = None,
        query_prefix: str | None = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.device = device
        self.query_prefix = (
            query_prefix if query_prefix is not None else QUERY_PREFIXES.get(model_name, "")
        )
        self._model = None

    @classmethod
    def from_config(cls, cfg: dict, **overrides):
        emb = cfg.get("embedding", {}) or {}
        kwargs = dict(
            model_name=emb.get("model", DEFAULT_MODEL),
            max_tokens=int(emb.get("max_tokens_per_chunk", 256)),
            batch_size=int(emb.get("batch_size", 64)),
            device=emb.get("device"),
            query_prefix=emb.get("query_prefix"),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_tokens > self._model.max_seq_length:
                raise ValueError(
                    f"max_tokens_per_chunk={self.max_tokens} exceeds {self.model_name}'s "
                    f"max_seq_length={self._model.max_seq_length}; lower it in configs/config.yaml"
                )
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    # ---- tokenizer-based length control (CLAUDE.md guardrail) -------------------------

    def count_tokens(self, text: str) -> int:
        """Content tokens, excluding [CLS]/[SEP] -- comparable to max_tokens as a budget."""
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str) -> str:
        """Hard-cap a chunk at max_tokens using the model's own tokenizer.

        Two special tokens are reserved so the encoded sequence including [CLS]/[SEP] still
        fits under the cap. If we didn't truncate here, sentence-transformers would silently
        truncate at max_seq_length instead -- same data loss, but invisible and at a different
        limit than the one configured.
        """
        ids = self.model.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_tokens - 2:
            return text
        kept = ids[: self.max_tokens - 2]
        return self.model.tokenizer.decode(kept, skip_special_tokens=True)

    def truncation_report(self, texts: list[str]) -> dict:
        """How hard the token cap is biting. Worth logging at index-build time: if a large
        share of *table_row* chunks truncate, numeric cells are being dropped and
        max_tokens_per_chunk needs raising before any retrieval number is trustworthy."""
        counts = [self.count_tokens(t) for t in texts]
        over = [c for c in counts if c > self.max_tokens - 2]
        return {
            "n_texts": len(texts),
            "n_truncated": len(over),
            "pct_truncated": round(100.0 * len(over) / len(texts), 2) if texts else 0.0,
            "max_tokens_cap": self.max_tokens,
            "p50_tokens": int(np.percentile(counts, 50)) if counts else 0,
            "p95_tokens": int(np.percentile(counts, 95)) if counts else 0,
            "max_tokens_seen": max(counts) if counts else 0,
        }

    # ---- encoding ---------------------------------------------------------------------

    def encode_chunks(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed passages. Truncated first, so the configured cap is the one that applies."""
        prepared = [self.truncate(t) for t in texts]
        return self._encode(prepared, show_progress)

    def encode_queries(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed queries (BGE instruction prefix applied). Also the entry point Month 4's
        attribution uses to batch-embed perturbed queries in one call."""
        prepared = [self.query_prefix + t for t in texts]
        return self._encode(prepared, show_progress)

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # inner product == cosine; see module docstring
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)


class HashEmbedder:
    """Deterministic offline stand-in with Embedder's interface: hashed bag-of-words into a
    fixed-dim space, L2-normalized. Exact-term overlap only -- no semantics whatsoever.

    Exists so tests (and a no-network demo) can exercise the whole index/retriever/eval path
    without downloading a model. Numbers produced with it are meaningless as retrieval
    quality; the eval script refuses to write results under a real dataset name when it is in
    use unless explicitly asked.
    """

    model_name = "hash-embedder-offline"

    def __init__(self, dim: int = 128, max_tokens: int = 256):
        self._dim = dim
        self.max_tokens = max_tokens
        self.query_prefix = ""

    @property
    def dim(self) -> int:
        return self._dim

    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\w+", text))

    def truncate(self, text: str) -> str:
        toks = re.findall(r"\w+", text)
        return " ".join(toks[: self.max_tokens])

    def truncation_report(self, texts: list[str]) -> dict:
        counts = [self.count_tokens(t) for t in texts]
        over = [c for c in counts if c > self.max_tokens]
        return {
            "n_texts": len(texts),
            "n_truncated": len(over),
            "pct_truncated": round(100.0 * len(over) / len(texts), 2) if texts else 0.0,
            "max_tokens_cap": self.max_tokens,
            "p50_tokens": int(np.percentile(counts, 50)) if counts else 0,
            "p95_tokens": int(np.percentile(counts, 95)) if counts else 0,
            "max_tokens_seen": max(counts) if counts else 0,
        }

    def encode_chunks(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", self.truncate(text).lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self._dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
