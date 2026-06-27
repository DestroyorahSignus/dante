"""Reciprocal Rank Fusion — DANTE_BUILD_PLAN.md §4.4.

RRF is score-distribution-agnostic (it uses ranks, not raw scores), so it fuses
legs whose scores live on different scales (BM25 in [0, 30], cosine in [0, 1])
without any normalization. ``k=60`` is the standard constant.
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_n: int = 200,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    ``RRF_score(d) = Σ_{r ∈ ranked_lists} 1 / (k + rank_r(d))``

    Args:
        ranked_lists: One ``[(doc_id, score)]`` list per retriever. Only the
            ordering matters; the per-leg scores are ignored.
        k: RRF constant (default 60).
        top_n: Number of fused results to return.

    Returns:
        Fused ``[(doc_id, rrf_score)]`` sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_n]
