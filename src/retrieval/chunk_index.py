"""
Chunk index: cached chunk embeddings + the two ways we search them.

**Why per-document retrieval is the default.** Both v1 datasets pose retrieval as
"find the supporting facts *inside this filing excerpt*": FinQA's `gold_inds` and TAT-QA's
`rel_paragraphs` are indices into a single document, and chunk_ids are only unique within a
doc_id (every FinQA doc has a `text_3`). So a question's candidate pool is its own document's
chunks, and Precision@k/Recall@k are computed there. `scope="corpus"` searches every chunk in
the index instead -- more like a deployed RAG system, kept available because Month 7 may want
that harder setting, but it is not the setting the gold labels were written for.

**The Month 4 hook.** `score()` is a single matrix multiply against embeddings that are already
in memory. Attribution perturbs a query N times, batch-embeds all N at once, and calls
`score()` once to get an (N x n_chunks) matrix. That is the whole reason chunk embeddings are
cached as a plain array and normalized at encode time. The anti-pattern the guardrails call
out -- re-running retrieval, or worse the generator, inside a perturbation loop -- should never
be necessary: if you find yourself wanting it, the primitive you actually want is here.

FAISS backs `scope="corpus"` (config: `retrieval.index_type: faiss_flat_ip`). Flat IP over
L2-normalized vectors is exact cosine search -- no approximation, no recall cliff to explain
away in the paper. It is rebuilt from the cached embeddings on load rather than serialized:
building a flat index is just a memcpy, and a persisted .index that silently disagrees with
embeddings.npy is a debugging afternoon nobody needs.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC / "ingestion"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from schema import Chunk  # noqa: E402

SCHEMA_VERSION = 1


@dataclass
class RetrievedChunk:
    """One retrieval hit. `score` is cosine similarity in [-1, 1] (inner product of
    L2-normalized vectors), `rank` is 0-based within this result list."""
    chunk: Chunk
    score: float
    rank: int

    def to_json(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "chunk_type": self.chunk.chunk_type,
            "text": self.chunk.text,
            "score": round(float(self.score), 6),
            "rank": self.rank,
        }


class ChunkIndex:
    """Chunks + their embeddings + a doc_id -> row-indices map, with exact search over either
    one document or the whole corpus."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, meta: dict | None = None):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {embeddings.shape[0]}"
            )
        self.chunks = chunks
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.meta = meta or {}
        self._doc_rows: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self._doc_rows.setdefault(c.doc_id, []).append(i)
        self._faiss = None

    # ---- construction / persistence ----------------------------------------------------

    @classmethod
    def build(cls, chunks: list[Chunk], embedder, show_progress: bool = False) -> "ChunkIndex":
        texts = [c.text for c in chunks]
        report = embedder.truncation_report(texts)
        embeddings = embedder.encode_chunks(texts, show_progress=show_progress)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "model_name": embedder.model_name,
            "dim": int(embeddings.shape[1]) if len(chunks) else 0,
            "max_tokens": getattr(embedder, "max_tokens", None),
            "n_chunks": len(chunks),
            "n_docs": len(set(c.doc_id for c in chunks)),
            "datasets": sorted(set(c.dataset for c in chunks)),
            "truncation": report,
        }
        return cls(chunks, embeddings, meta)

    def save(self, out_dir) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", self.embeddings)
        with open(out_dir / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_json()) + "\n")
        with open(out_dir / "index_meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)
        return out_dir

    @classmethod
    def load(cls, index_dir, expect_model: str | None = None) -> "ChunkIndex":
        """Reload a built index. `expect_model` guards the failure mode that produces
        plausible-looking garbage: querying with one embedding model against chunk vectors
        built by another. Scores stay in range, ranking is nonsense."""
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "index_meta.json").read_text())
        if expect_model and meta.get("model_name") != expect_model:
            raise ValueError(
                f"index at {index_dir} was built with '{meta.get('model_name')}' but the "
                f"current config uses '{expect_model}' -- rebuild the index (--rebuild) "
                f"instead of mixing embedding spaces"
            )
        embeddings = np.load(index_dir / "embeddings.npy")
        chunks = []
        with open(index_dir / "chunks.jsonl") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))
        return cls(chunks, embeddings, meta)

    # ---- lookups -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_rows.keys())

    def rows_for_doc(self, doc_id: str) -> list[int]:
        if doc_id not in self._doc_rows:
            raise KeyError(f"doc_id {doc_id!r} not in index ({len(self._doc_rows)} docs)")
        return self._doc_rows[doc_id]

    def chunks_for_doc(self, doc_id: str) -> list[Chunk]:
        return [self.chunks[i] for i in self.rows_for_doc(doc_id)]

    # ---- scoring -----------------------------------------------------------------------

    def score(self, query_vecs: np.ndarray, doc_id: str | None = None) -> np.ndarray:
        """Cosine similarity of every query against every candidate chunk, in one matmul.

        Returns (n_queries, n_candidates). Candidates are `chunks_for_doc(doc_id)` in order
        when doc_id is given, otherwise all chunks in index order.

        This is the primitive Month 4 attribution is built on: embed all query perturbations
        in one `encode_queries` call, then one `score()` call here. No re-indexing, no
        generator calls, no per-perturbation loop over the retriever.
        """
        query_vecs = np.atleast_2d(np.asarray(query_vecs, dtype=np.float32))
        if doc_id is None:
            return query_vecs @ self.embeddings.T
        rows = self.rows_for_doc(doc_id)
        return query_vecs @ self.embeddings[rows].T

    def search(
        self, query_vecs: np.ndarray, k: int = 5, doc_id: str | None = None
    ) -> list[list[RetrievedChunk]]:
        """Top-k per query. Per-document search goes through `score()` (a doc has tens of
        chunks -- a matmul beats index bookkeeping); corpus search goes through FAISS."""
        query_vecs = np.atleast_2d(np.asarray(query_vecs, dtype=np.float32))
        if doc_id is not None:
            sims = self.score(query_vecs, doc_id=doc_id)
            candidates = self.chunks_for_doc(doc_id)
            results = []
            for row in sims:
                k_eff = min(k, len(candidates))
                # argpartition for the top-k set, then sort just those k
                top = np.argpartition(-row, k_eff - 1)[:k_eff] if k_eff < len(row) else np.arange(len(row))
                top = top[np.argsort(-row[top])]
                results.append([
                    RetrievedChunk(chunk=candidates[j], score=float(row[j]), rank=r)
                    for r, j in enumerate(top)
                ])
            return results

        scores, idxs = self._faiss_index().search(query_vecs, min(k, len(self.chunks)))
        return [
            [
                RetrievedChunk(chunk=self.chunks[j], score=float(s), rank=r)
                for r, (j, s) in enumerate(zip(row_idx, row_score))
                if j >= 0
            ]
            for row_idx, row_score in zip(idxs, scores)
        ]

    def _faiss_index(self):
        if self._faiss is None:
            import faiss

            index = faiss.IndexFlatIP(self.embeddings.shape[1])
            index.add(self.embeddings)
            self._faiss = index
        return self._faiss
