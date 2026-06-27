"""ColBERT late-interaction reranker — DANTE_BUILD_PLAN.md §4.5.

Uses AnswerDotAI's **`rerankers`** library to load ``answerdotai/answerai-colbert-small-v1``
and MaxSim-rerank the fused candidates. We use `rerankers` (purpose-built for this model,
dependency-light) rather than pylate, which couples to sentence-transformers' internal API
and broke on the pinned ST 5.6 (`generate_model_card` import error → silent no-op).

CRITICAL (plan D0.2): the ablation must NEVER crash because of ColBERT. Any failure
— import error, model load failure, scoring error — logs a warning and returns the
candidates in their INCOMING order (identity fallback), so the "+ ColBERT" row simply
degrades to the fused ranking instead of taking the whole run down.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("dante.colbert")

DEFAULT_COLBERT = "answerdotai/answerai-colbert-small-v1"

# Process-level cache so we load the ColBERT model once, not per query.
_MODEL_CACHE: dict = {}


def _candidate_id(c) -> str:
    """Extract a product id from a candidate (dict or (id, ...) tuple or str)."""
    if isinstance(c, dict):
        return str(c.get("product_id") or c.get("id") or c.get("doc_id"))
    if isinstance(c, (tuple, list)) and c:
        return str(c[0])
    return str(c)


def _candidate_text(c) -> str:
    """Extract the text to score from a candidate dict (best-effort)."""
    if isinstance(c, dict):
        return str(c.get("product_text") or c.get("text") or c.get("document") or "")
    return str(c)


def _load_model(model_name: str):
    """Load (and cache) a `rerankers` ColBERT model, or return None on any failure."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from rerankers import Reranker  # type: ignore

        model = Reranker(model_name, model_type="colbert", verbose=0)
        _MODEL_CACHE[model_name] = model
        return model
    except Exception as exc:  # import error, download failure, etc.
        logger.warning("ColBERT load failed (%s) — using identity fallback.", exc)
        _MODEL_CACHE[model_name] = None
        return None


def _result_index(r) -> int:
    """Pull the (integer) doc_id back out of a rerankers Result, across API variants."""
    doc = getattr(r, "document", None)
    if doc is not None and getattr(doc, "doc_id", None) is not None:
        return int(doc.doc_id)
    if getattr(r, "doc_id", None) is not None:
        return int(r.doc_id)
    return int(getattr(r, "id", 0))


def colbert_rerank(query: str, candidates: list, top_k: int = 20, model=None):
    """MaxSim-rerank ``candidates`` for ``query``; identity fallback on any failure.

    Args:
        query: The search query.
        candidates: Candidate items — dicts with ``product_text`` (preferred),
            ``(product_id, ...)`` tuples, or strings. Order is the fused ranking.
        top_k: Number of reranked items to return.
        model: An optional preloaded model (name str or a `rerankers.Reranker`). If
            None, the default ``answerai-colbert-small-v1`` is loaded + cached.

    Returns:
        A list of the (up to) ``top_k`` candidates in reranked order. If ColBERT is
        unavailable for ANY reason, returns the first ``top_k`` candidates in their
        incoming order (identity fallback) — never raises.
    """
    if not candidates:
        return []

    try:
        if isinstance(model, str) or model is None:
            model = _load_model(model or DEFAULT_COLBERT)

        if model is None:
            return candidates[:top_k]  # graceful identity fallback

        texts = [_candidate_text(c) for c in candidates]
        doc_ids = list(range(len(candidates)))
        # rerankers: doc_ids are indices into `candidates`; results come back sorted.
        ranked = model.rank(query=str(query), docs=texts, doc_ids=doc_ids)
        results = getattr(ranked, "results", ranked)
        order = [_result_index(r) for r in results]
        # Keep only valid indices; backfill any dropped ones in original order.
        seen = set()
        out_idx = [i for i in order if 0 <= i < len(candidates) and not (i in seen or seen.add(i))]
        if len(out_idx) < len(candidates):
            out_idx += [i for i in doc_ids if i not in seen]
        return [candidates[i] for i in out_idx[:top_k]]
    except Exception as exc:
        logger.warning("ColBERT rerank failed (%s) — identity fallback.", exc)
        return candidates[:top_k]
