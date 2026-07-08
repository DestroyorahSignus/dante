"""ModernBERT bi-encoder dense retrieval — DANTE_BUILD_PLAN.md §4.2.

Inference helpers (``build_dense_index`` / ``dense_search``) build a FAISS
``IndexFlatIP`` over L2-normalized embeddings (cosine similarity — RISK R7: BOTH
docs and queries are normalized).

``train_biencoder`` is a thin wrapper documenting that the LIVE, Modal-validated
trainer is ``modal_train.py::train_biencoder``. It is re-exported here only so the
public API in ``dante/__init__.py`` resolves and a consumer has a single import
surface. Run real training via ``modal run modal_train.py --stage train``.
"""
from __future__ import annotations


def build_dense_index(model, product_ids: list[str], texts: list[str], batch_size: int = 256):
    """Encode the catalog and build a FAISS ``IndexFlatIP`` (cosine via L2-norm).

    Args:
        model: A loaded ``SentenceTransformer`` (passed in, so consumers like SPARDA
            can hand DANTE's trained encoder straight through — §2.1).
        product_ids: Parallel product ids.
        texts: Parallel product texts.
        batch_size: Encode batch size.

    Returns:
        ``(faiss_index, product_ids)``.
    """
    import faiss

    if len(product_ids) != len(texts):
        raise ValueError("product_ids and texts must be the same length")
    emb = model.encode(
        list(texts), batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype("float32")
    faiss.normalize_L2(emb)  # RISK R7: normalize docs before IndexFlatIP
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, list(product_ids)


def dense_search(model, index, product_ids: list[str], query: str, top_k: int = 1000):
    """Encode the query (L2-normed) and search the FAISS index.

    Returns the top-k ``(product_id, score)`` pairs.
    """
    import faiss

    q = model.encode([str(query)], convert_to_numpy=True,
                     normalize_embeddings=False).astype("float32")
    faiss.normalize_L2(q)  # RISK R7: normalize the query too
    k = min(top_k, len(product_ids))
    scores, idx = index.search(q, k)
    return [(product_ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


def train_biencoder(train_pairs=None, val_pairs=None, config=None):
    """Re-export shim for the public API — the live trainer lives in modal_train.py.

    DANTE's bi-encoder fine-tune is the single GPU job and is implemented + validated
    as the Modal function ``modal_train.py::train_biencoder`` (MNRL on ESCI pairs,
    A100, W&B logging). This package-level symbol exists so ``dante/__init__.py``
    resolves and so a consumer importing ``from dante import train_biencoder`` gets a
    clear pointer rather than an ImportError.

    Run training with:
        ``modal run modal_train.py --stage train``

    Raises:
        NotImplementedError: always — directs the caller to the Modal entrypoint.
    """
    raise NotImplementedError(
        "The live bi-encoder trainer is modal_train.py::train_biencoder "
        "(run: `modal run modal_train.py --stage train`). This package-level "
        "train_biencoder is a re-export shim for the public API surface only."
    )
