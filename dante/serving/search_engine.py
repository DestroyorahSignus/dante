"""Unified DANTE search engine — DANTE_BUILD_PLAN.md §4.6.

Full pipeline: query → 3-leg retrieve (dense + sparse + lexical) top-1000 → RRF fuse
(k=60) top-200 → ColBERT rerank top-k. Every component is decoupled via config; a
consumer constructs and runs DANTE without importing anything else.

The per-leg helpers (``dense_only`` / ``bm25_only`` / ``splade_only`` / ``fused``)
return ranked product-id lists and are reused directly by the eval ablation, so every
ablation row shares the SAME retrieval code as production.
"""
from __future__ import annotations

import json
import os

from ..models.biencoder import dense_search
from ..models.bm25 import BM25Index
from ..models.colbert_reranker import colbert_rerank
from ..models.fusion import reciprocal_rank_fusion
from ..models.splade import DEFAULT_MODEL as DEFAULT_SPLADE_MODEL
from ..models.splade import SpladeEncoder


def _cfg(config, *keys, default=None):
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


class DanteSearchEngine:
    """The full DANTE pipeline: query → fused → reranked."""

    def __init__(self, config):
        """Load indices + models from the paths in ``config`` (§4.6)."""
        import faiss
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        self.config = config
        index_dir = _cfg(config, "serving", "index_dir", default="/artifacts/index")
        catalog_path = _cfg(config, "serving", "catalog_path",
                            default="/artifacts/data/catalog.parquet")
        biencoder_path = _cfg(config, "biencoder", "path", default="/artifacts/biencoder_final")
        splade_model = _cfg(config, "splade", "model", default=DEFAULT_SPLADE_MODEL)
        splade_maxlen = _cfg(config, "splade", "max_length", default=256)

        self.rrf_k = _cfg(config, "serving", "rrf_k", default=60)
        self.top_n = _cfg(config, "serving", "top_n", default=200)
        self.leg_top_k = _cfg(config, "serving", "leg_top_k", default=1000)
        self.colbert_model = _cfg(config, "colbert", "model",
                                  default="answerdotai/answerai-colbert-small-v1")

        # Single product-id list shared by all legs.
        with open(os.path.join(index_dir, "product_ids.json")) as f:
            self.product_ids = json.load(f)

        # Dense leg.
        self.biencoder = SentenceTransformer(biencoder_path)
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "dense.faiss"))

        # Lexical leg.
        self.bm25 = BM25Index().load(os.path.join(index_dir, "bm25.pkl"))

        # Sparse leg.
        self.splade = SpladeEncoder(model_name=splade_model, max_length=splade_maxlen)
        self.splade.load(os.path.join(index_dir, "splade.npz"))

        # Product metadata DB (id -> {product_id, product_text}) for reranking.
        catalog = pd.read_parquet(catalog_path)
        self.product_db = {
            str(r.product_id): {"product_id": str(r.product_id),
                                "product_text": str(r.product_text)}
            for r in catalog.itertuples(index=False)
        }

    # -- per-leg helpers (return [(id, score)] ranked lists) -----------------
    def _dense(self, query: str, top_k: int):
        return dense_search(self.biencoder, self.faiss_index, self.product_ids, query, top_k)

    def _bm25(self, query: str, top_k: int):
        return self.bm25.search(query, top_k)

    def _splade(self, query: str, top_k: int):
        return self.splade.search(query, top_k)

    # -- ablation-facing helpers (return ranked product-id lists) ------------
    def dense_only(self, query: str, top_k: int | None = None) -> list[str]:
        return [pid for pid, _ in self._dense(query, top_k or self.leg_top_k)]

    def bm25_only(self, query: str, top_k: int | None = None) -> list[str]:
        return [pid for pid, _ in self._bm25(query, top_k or self.leg_top_k)]

    def splade_only(self, query: str, top_k: int | None = None) -> list[str]:
        return [pid for pid, _ in self._splade(query, top_k or self.leg_top_k)]

    def fused(self, query: str, legs=("dense", "bm25", "splade"),
              top_n: int | None = None) -> list[str]:
        """RRF-fuse a subset of legs; returns ranked product ids (length <= top_n)."""
        leg_fns = {"dense": self._dense, "bm25": self._bm25, "splade": self._splade}
        ranked = [leg_fns[leg](query, self.leg_top_k) for leg in legs]
        fused = reciprocal_rank_fusion(ranked, k=self.rrf_k, top_n=top_n or self.top_n)
        return [pid for pid, _ in fused]

    # -- full pipeline -------------------------------------------------------
    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """3-leg retrieve → RRF fuse → ColBERT rerank → top-k product dicts (§4.6)."""
        fused_ids = self.fused(query, legs=("dense", "bm25", "splade"), top_n=self.top_n)
        candidates = [self.product_db[pid] for pid in fused_ids if pid in self.product_db]
        return colbert_rerank(query, candidates, top_k=top_k, model=self.colbert_model)
