"""BM25 lexical retrieval — DANTE_BUILD_PLAN.md §4.1.

A thin wrapper over ``rank_bm25.BM25Okapi`` that keeps the parallel ``doc_ids``
list so search can return ``(product_id, score)`` pairs. Build/search/save/load.
"""
from __future__ import annotations

import pickle


class BM25Index:
    """BM25 (Okapi) lexical index over a product catalog.

    Tokenization is a simple lowercase whitespace split, matched on the query side
    so the lexical leg stays symmetric with how the catalog was indexed.
    """

    def __init__(self) -> None:
        self.bm25 = None
        self.doc_ids: list[str] = []

    def build(self, product_ids: list[str], product_texts: list[str]) -> "BM25Index":
        """Tokenize and build the BM25 index.

        Args:
            product_ids: Parallel list of product ids.
            product_texts: Parallel list of product texts to index.

        Returns:
            self (so callers can chain ``BM25Index().build(...)``).
        """
        from rank_bm25 import BM25Okapi

        if len(product_ids) != len(product_texts):
            raise ValueError(
                f"product_ids ({len(product_ids)}) and product_texts "
                f"({len(product_texts)}) must be the same length"
            )
        tokenized = [str(text).lower().split() for text in product_texts]
        self.bm25 = BM25Okapi(tokenized)
        self.doc_ids = list(product_ids)
        return self

    def search(self, query: str, top_k: int = 1000) -> list[tuple[str, float]]:
        """Return the top-k ``(product_id, score)`` pairs for a query."""
        if self.bm25 is None:
            raise RuntimeError("BM25Index is empty — call build() or load() first.")
        import numpy as np

        scores = self.bm25.get_scores(str(query).lower().split())
        k = min(top_k, len(self.doc_ids))
        # argpartition for the top-k, then sort just those k (cheaper than full sort).
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

    def save(self, path: str) -> None:
        """Pickle ``(bm25, doc_ids)`` to ``path``."""
        with open(path, "wb") as f:
            pickle.dump((self.bm25, self.doc_ids), f)

    def load(self, path: str) -> "BM25Index":
        """Load ``(bm25, doc_ids)`` from a pickle written by :meth:`save`."""
        with open(path, "rb") as f:
            self.bm25, self.doc_ids = pickle.load(f)
        return self
