"""Rerankers (ColBERT late-interaction + cross-encoder) — DANTE_BUILD_PLAN.md §4.5.

Uses AnswerDotAI's **`rerankers`** library to load ``answerdotai/answerai-colbert-small-v1``
and MaxSim-rerank the fused candidates. We use `rerankers` (purpose-built for this model,
dependency-light) rather than pylate, which couples to sentence-transformers' internal API
and broke on the pinned ST 5.6 (`generate_model_card` import error → silent no-op).

The generic ``rerank(...)`` also serves cross-encoder models through the same library
(e.g. ``BAAI/bge-reranker-base`` with ``model_type="cross-encoder"`` — ungated, works on
the pinned transformers 4.57.6), so the ablation can compare late-interaction vs CE
reranking over identical fused candidates.

CRITICAL (plan D0.2): the ablation must NEVER crash because of a reranker. Any failure
— import error, model load failure, scoring error — logs a warning and returns the
candidates in their INCOMING order (identity fallback), so a rerank row simply
degrades to the fused ranking instead of taking the whole run down.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("dante.colbert")

DEFAULT_COLBERT = "answerdotai/answerai-colbert-small-v1"
DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-base"

# Process-level cache so we load each reranker model once, not per query.
# Keyed by (model_name, model_type) — the same checkpoint could in principle be
# loaded under different rerankers model_types.
_MODEL_CACHE: dict = {}


def _candidate_text(c) -> str:
    """Extract the text to score from a candidate dict (best-effort)."""
    if isinstance(c, dict):
        return str(c.get("product_text") or c.get("text") or c.get("document") or "")
    return str(c)


def _load_model(model_name: str, model_type: str = "colbert"):
    """Load (and cache) a `rerankers` model, or return None on any failure."""
    key = (model_name, model_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from rerankers import Reranker  # type: ignore

        model = Reranker(model_name, model_type=model_type, verbose=0)
        _MODEL_CACHE[key] = model
        return model
    except Exception as exc:  # import error, download failure, etc.
        logger.warning("Reranker load failed (%s/%s: %s) — using identity fallback.",
                       model_name, model_type, exc)
        _MODEL_CACHE[key] = None
        return None


def _result_index(r) -> int:
    """Pull the (integer) doc_id back out of a rerankers Result, across API variants."""
    doc = getattr(r, "document", None)
    if doc is not None and getattr(doc, "doc_id", None) is not None:
        return int(doc.doc_id)
    if getattr(r, "doc_id", None) is not None:
        return int(r.doc_id)
    return int(getattr(r, "id", 0))


def rerank(query: str, candidates: list, top_k: int = 20,
           model_name: str = DEFAULT_COLBERT, model_type: str = "colbert",
           model=None):
    """Rerank ``candidates`` for ``query`` with any `rerankers` model; identity fallback.

    Args:
        query: The search query.
        candidates: Candidate items — dicts with ``product_text`` (preferred),
            ``(product_id, ...)`` tuples, or strings. Order is the fused ranking.
        top_k: Number of reranked items to return.
        model_name: HF checkpoint to load via `rerankers` (cached per process).
        model_type: `rerankers` model type — ``"colbert"`` (MaxSim late-interaction)
            or ``"cross-encoder"`` (e.g. ``BAAI/bge-reranker-base``).
        model: An optional preloaded model (name str or a `rerankers.Reranker`);
            overrides ``model_name`` when given.

    Returns:
        A list of the (up to) ``top_k`` candidates in reranked order. If the model is
        unavailable for ANY reason, returns the first ``top_k`` candidates in their
        incoming order (identity fallback) — never raises.
    """
    if not candidates:
        return []

    try:
        if isinstance(model, str):
            model = _load_model(model, model_type)
        elif model is None:
            model = _load_model(model_name, model_type)

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
        logger.warning("Rerank failed (%s/%s: %s) — identity fallback.",
                       model_name, model_type, exc)
        return candidates[:top_k]


def colbert_rerank(query: str, candidates: list, top_k: int = 20, model=None):
    """MaxSim-rerank ``candidates`` for ``query``; identity fallback on any failure.

    Backward-compatible wrapper over the generic ``rerank`` (model_type="colbert").
    If ``model`` is None, the default ``answerai-colbert-small-v1`` is loaded + cached.
    """
    return rerank(query, candidates, top_k=top_k,
                  model_name=DEFAULT_COLBERT, model_type="colbert", model=model)


def ce_rerank(query: str, candidates: list, top_k: int = 20, model=None):
    """Cross-encoder rerank (default ``BAAI/bge-reranker-base``); identity fallback.

    Sibling of ``colbert_rerank`` over the same generic ``rerank`` — same candidate
    formats, same never-crash contract, same per-process model cache.
    """
    return rerank(query, candidates, top_k=top_k,
                  model_name=DEFAULT_CROSS_ENCODER, model_type="cross-encoder",
                  model=model)
