"""Hard-negative mining — DANTE_BUILD_PLAN.md §4.2.

IMPLEMENTED as the ``mine`` stage in ``modal_train.py`` (repo root), which follows
the §4.2 DENSE-FIRST recipe: batched-FAISS mining over the FULL catalog with
``sentence_transformers.util.mine_hard_negatives`` (range_min/margin guards against
unlabeled positives), writing ``/artifacts/data/train_hn`` n-tuples that MNRL
consumes natively. It runs on the volume artifacts inside Modal, so it lives with
the other stages rather than here.

This module's BM25 variant (the optional lexical-diversity supplement) remains a
documented stub — §4.2 explicitly warns against the brute-force rank_bm25 loop.
"""


def mine_hard_negatives(queries, products, bm25_index, biencoder=None, num_negatives=5):
    """OPTIONAL BM25 lexical-diversity supplement (NOT the primary path).

    The primary dense-first mining is `modal_train.py::mine`. If ever built, this
    would: for each TRAIN query only (never the full cross-product), take the BM25
    top-100, filter out labeled positives (E/S), and sample a few negatives.

    Args:
        queries: Query records.
        products: Product catalog records.
        bm25_index: A built BM25Index.
        biencoder: Optional dense retriever for cross-mining (after first epoch).
        num_negatives: Hard negatives per query.

    Returns:
        Mapping of query_id → list of hard-negative product_ids.
    """
    raise NotImplementedError(
        "BM25 supplement not built — use the dense-first `mine` stage in "
        "modal_train.py (DANTE_BUILD_PLAN.md §4.2)"
    )
