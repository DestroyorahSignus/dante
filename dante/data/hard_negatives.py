"""Hard-negative mining — DANTE_BUILD_PLAN.md §4.2."""


def mine_hard_negatives(queries, products, bm25_index, biencoder=None, num_negatives=5):
    """Mine hard negatives for bi-encoder / SPLADE training.

    For each query, get BM25 top-100, filter out positives (E/S labels), take the top-K
    remaining as hard negatives. Optionally also mine from a first-pass dense retriever
    (cross-mine): combine e.g. 3 from BM25 + 2 from dense = 5 hard negatives per query.
    Mine BM25 and dense negatives separately — do not mix them in the same batch.

    Args:
        queries: Query records.
        products: Product catalog records.
        bm25_index: A built BM25Index.
        biencoder: Optional dense retriever for cross-mining (after first epoch).
        num_negatives: Hard negatives per query.

    Returns:
        Mapping of query_id → list of hard-negative product_ids.
    """
    raise NotImplementedError("TODO: see DANTE_BUILD_PLAN.md §4.2")
