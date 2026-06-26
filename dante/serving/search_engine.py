"""Unified DANTE search engine — DANTE_BUILD_PLAN.md §4.6.

Full pipeline: query → 3-leg retrieve (dense + sparse + lexical) → RRF fuse → ColBERT
rerank. Every component is decoupled via config; a consumer constructs and runs DANTE
without importing anything else.
"""


class DanteSearchEngine:
    """The full DANTE pipeline: query → fused → reranked."""

    def __init__(self, config):
        """Construct the engine, loading each component from paths in config.

        Loads the BM25 index, the ModernBERT bi-encoder + FAISS index, the SPLADE
        encoder, the ColBERT reranker, and the product metadata DB.

        Args:
            config: Mapping with biencoder_path, splade_path, colbert_path, index
                paths, and product DB location.
        """
        raise NotImplementedError("TODO: see DANTE_BUILD_PLAN.md §4.6")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Full pipeline: 3-leg retrieve → RRF fuse → ColBERT rerank.

        Args:
            query: The search query.
            top_k: Number of final reranked results to return.

        Returns:
            Ranked list of product dicts with scores and retriever attribution.
        """
        raise NotImplementedError("TODO: see DANTE_BUILD_PLAN.md §4.6")
