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
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists with (optionally weighted) Reciprocal Rank Fusion.

    ``RRF_score(d) = Σ_{r ∈ ranked_lists} w_r / (k + rank_r(d))``

    Args:
        ranked_lists: One ``[(doc_id, score)]`` list per retriever. Only the
            ordering matters; the per-leg scores are ignored.
        k: RRF constant (default 60).
        top_n: Number of fused results to return.
        weights: Optional per-list weights (parallel to ``ranked_lists``). Each
            list's ``1/(k+rank)`` contribution is multiplied by its weight, so a
            noisier leg (e.g. BM25) can be down-weighted instead of dropped.
            ``None`` (default) = uniform weights of 1.0 — identical to classic
            RRF, so existing callers are unaffected.

    Returns:
        Fused ``[(doc_id, rrf_score)]`` sorted by descending RRF score.
    """
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights has {len(weights)} entries but there are "
            f"{len(ranked_lists)} ranked lists"
        )
    scores: dict[str, float] = {}
    for li, ranked_list in enumerate(ranked_lists):
        w = 1.0 if weights is None else float(weights[li])
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_n]
