"""
Retriever -- the B1 baseline's retrieval half, and the single object every later module
talks to when it needs "what would the retriever score this?".

Deliberately thin: config in, embedder + ChunkIndex held together, top-k out. No reranker,
no BM25 hybrid, no query rewriting (v1 scope in CLAUDE.md is one dense retriever; hybrid
retrieval is a stretch goal, not a baseline). Keeping B1 plain is the point -- it's the
comparison floor for B2/B3, so anything clever added here quietly weakens the contribution.

Downstream contracts:
  - generation (B1)  -> `retrieve()`
  - attribution (M4) -> `score_query_variants()`, one batched matmul, never a re-retrieval loop
  - demo             -> `Retriever.from_config(...)` once, cached with st.cache_resource
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / "ingestion"), str(_SRC / "retrieval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chunk_index import ChunkIndex, RetrievedChunk  # noqa: E402
from config_utils import get as cfg_get, load_config  # noqa: E402
from embedder import Embedder  # noqa: E402
from schema import Chunk  # noqa: E402


class Retriever:
    def __init__(self, embedder, index: ChunkIndex, top_k: int = 5, scope: str = "document"):
        if scope not in ("document", "corpus"):
            raise ValueError(f"scope must be 'document' or 'corpus', got {scope!r}")
        self.embedder = embedder
        self.index = index
        self.top_k = top_k
        self.scope = scope

    # ---- construction -------------------------------------------------------------------

    @classmethod
    def build(cls, chunks: list[Chunk], cfg: dict | None = None, embedder=None,
              show_progress: bool = False) -> "Retriever":
        cfg = cfg or load_config()
        embedder = embedder or Embedder.from_config(cfg)
        index = ChunkIndex.build(chunks, embedder, show_progress=show_progress)
        return cls(
            embedder=embedder,
            index=index,
            top_k=int(cfg_get(cfg, "retrieval.top_k", 5)),
            scope=cfg_get(cfg, "retrieval.scope", "document"),
        )

    @classmethod
    def from_config(cls, config_path=None, index_dir=None, embedder=None) -> "Retriever":
        """Load a previously built index (default location: data/processed/index/). Raises if
        the index was built with a different embedding model than the config now names."""
        cfg = load_config(config_path)
        embedder = embedder or Embedder.from_config(cfg)
        from config_utils import resolve_path

        index_dir = Path(index_dir) if index_dir else resolve_path(
            cfg_get(cfg, "retrieval.index_dir", "data/processed/index")
        )
        index = ChunkIndex.load(index_dir, expect_model=embedder.model_name)
        return cls(
            embedder=embedder,
            index=index,
            top_k=int(cfg_get(cfg, "retrieval.top_k", 5)),
            scope=cfg_get(cfg, "retrieval.scope", "document"),
        )

    # ---- retrieval ----------------------------------------------------------------------

    def retrieve(self, question: str, doc_id: str | None = None, k: int | None = None
                 ) -> list[RetrievedChunk]:
        return self.retrieve_batch([question], [doc_id], k=k)[0]

    def retrieve_batch(self, questions: list[str], doc_ids: list[str | None] | None = None,
                       k: int | None = None, show_progress: bool = False
                       ) -> list[list[RetrievedChunk]]:
        """Embeds all questions in one batch. doc_ids may be None (corpus scope) or a
        per-question document to restrict the candidate pool to.

        Note the per-doc case still runs one search per distinct doc_id -- questions can't
        share a candidate pool -- but they all share the single embedding pass, which is the
        expensive part.
        """
        k = k or self.top_k
        if not questions:
            return []
        doc_ids = doc_ids if doc_ids is not None else [None] * len(questions)
        if len(doc_ids) != len(questions):
            raise ValueError("questions and doc_ids must be the same length")

        qvecs = self.embedder.encode_queries(questions, show_progress=show_progress)

        if self.scope == "corpus":
            return self.index.search(qvecs, k=k, doc_id=None)

        out = []
        for vec, doc_id in zip(qvecs, doc_ids):
            if doc_id is None:
                raise ValueError(
                    "scope='document' needs a doc_id per question; pass doc_ids= or set "
                    "retrieval.scope: corpus in configs/config.yaml"
                )
            out.append(self.index.search(vec[None, :], k=k, doc_id=doc_id)[0])
        return out

    # ---- Month 4 hook -------------------------------------------------------------------

    def score_query_variants(self, variants: list[str], doc_id: str | None = None
                             ) -> tuple[np.ndarray, list[Chunk]]:
        """Score N query variants against the candidate chunks in ONE embedding pass and ONE
        matmul. Returns (scores of shape (N, n_candidates), candidate chunks in column order).

        This is the intended entry point for perturbation-based retrieval attribution
        (Contribution 1): build the occlusion/surrogate variants of a query, hand the whole
        list here, and diff the score matrix rows against the unperturbed row. Nothing in
        this path touches the generator or rebuilds the index -- which is exactly the
        compute-exhausting anti-pattern the guardrails forbid.
        """
        qvecs = self.embedder.encode_queries(variants)
        scores = self.index.score(qvecs, doc_id=doc_id)
        candidates = (
            self.index.chunks_for_doc(doc_id) if doc_id is not None else self.index.chunks
        )
        return scores, candidates
