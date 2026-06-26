# dante/__init__.py — the ONLY surface other repos import
from .serving.search_engine import DanteSearchEngine   # full pipeline: query → fused → reranked
from .models.biencoder import train_biencoder, build_dense_index, dense_search
from .models.splade import SpladeEncoder
from .models.bm25 import BM25Index
from .models.colbert_reranker import colbert_rerank
from .models.fusion import reciprocal_rank_fusion

__all__ = [
    "DanteSearchEngine", "train_biencoder", "build_dense_index", "dense_search",
    "SpladeEncoder", "BM25Index", "colbert_rerank", "reciprocal_rank_fusion",
]
