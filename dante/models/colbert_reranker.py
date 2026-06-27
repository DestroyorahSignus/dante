"""ColBERT late-interaction reranker — DANTE_BUILD_PLAN.md §4.5.

Uses **pylate** (modern, ST-native ColBERT; lighter than ragatouille and friendlier
to recent torch/transformers) to load ``answerdotai/answerai-colbert-small-v1`` and
MaxSim-rerank the fused candidates.

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
    """Load (and cache) a pylate ColBERT model, or return None on any failure."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from pylate import models as pylate_models  # type: ignore

        model = pylate_models.ColBERT(model_name_or_path=model_name)
        _MODEL_CACHE[model_name] = model
        return model
    except Exception as exc:  # import error, download failure, etc.
        logger.warning("ColBERT load failed (%s) — using identity fallback.", exc)
        _MODEL_CACHE[model_name] = None
        return None


def colbert_rerank(query: str, candidates: list, top_k: int = 20, model=None):
    """MaxSim-rerank ``candidates`` for ``query``; identity fallback on any failure.

    Args:
        query: The search query.
        candidates: Candidate items — dicts with ``product_text`` (preferred),
            ``(product_id, ...)`` tuples, or strings. Order is the fused ranking.
        top_k: Number of reranked items to return.
        model: An optional preloaded pylate ColBERT model (name or instance). If
            None, the default ``answerai-colbert-small-v1`` is loaded + cached.

    Returns:
        A list of the (up to) ``top_k`` candidates in reranked order. If ColBERT is
        unavailable for ANY reason, returns the first ``top_k`` candidates in their
        incoming order (identity fallback) — never raises.
    """
    if not candidates:
        return []

    try:
        if isinstance(model, str):
            model = _load_model(model)
        elif model is None:
            model = _load_model(DEFAULT_COLBERT)

        if model is None:
            return candidates[:top_k]  # graceful identity fallback

        import torch

        texts = [_candidate_text(c) for c in candidates]
        # pylate: encode query (is_query=True) and docs, then MaxSim score.
        q_emb = model.encode([str(query)], is_query=True, convert_to_tensor=True)
        d_emb = model.encode(texts, is_query=False, convert_to_tensor=True)

        try:
            from pylate import scores as pylate_scores  # type: ignore

            sims = pylate_scores.colbert_scores(q_emb, d_emb)  # [1, n_docs]
            scores = sims[0].tolist()
        except Exception:
            # Manual MaxSim: Σ_i max_j (q_i · d_j), per document.
            qv = q_emb[0]  # [q_tokens, dim]
            scores = []
            for d in d_emb:  # d: [d_tokens, dim]
                sim = qv @ d.T            # [q_tokens, d_tokens]
                scores.append(float(sim.max(dim=1).values.sum()))

        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order[:top_k]]
    except Exception as exc:
        logger.warning("ColBERT rerank failed (%s) — identity fallback.", exc)
        return candidates[:top_k]
