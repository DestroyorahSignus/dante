"""Build FAISS + BM25 + SPLADE indices — DANTE_BUILD_PLAN.md §4.6 / §8.

Loads the trained bi-encoder + the catalog parquet and persists three indices to the
config-driven ``serving.index_dir`` (default ``/artifacts/index``):

  * dense.faiss          — FAISS IndexFlatIP over L2-normalized ModernBERT embeddings
  * product_ids.json     — the parallel product-id list (one source of truth)
  * bm25.pkl             — pickled BM25Okapi + doc_ids
  * splade.npz (+ .ids)  — CSR doc-term matrix + doc_ids

All three indices share the SAME catalog row order / product-id list.
"""
from __future__ import annotations

import json
import os

from ..models.bm25 import BM25Index
from ..models.biencoder import build_dense_index
from ..models.splade import SpladeEncoder


def _cfg(config, *keys, default=None):
    """Nested config getter: _cfg(config, 'serving', 'index_dir')."""
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def build_indices(config) -> dict:
    """Build the dense (FAISS), BM25 and SPLADE indices and save them to disk.

    Args:
        config: Parsed ``configs/default.yaml`` mapping.

    Returns:
        A dict of the written artifact paths.
    """
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    catalog_path = _cfg(config, "serving", "catalog_path", default="/artifacts/data/catalog.parquet")
    index_dir = _cfg(config, "serving", "index_dir", default="/artifacts/index")
    biencoder_path = _cfg(config, "biencoder", "path", default="/artifacts/biencoder_final")
    splade_model = _cfg(config, "splade", "model", default="naver/splade-v3")
    splade_maxlen = _cfg(config, "splade", "max_length", default=256)

    os.makedirs(index_dir, exist_ok=True)

    print(f"[index] loading catalog: {catalog_path}")
    catalog = pd.read_parquet(catalog_path)
    product_ids = catalog["product_id"].astype(str).tolist()
    texts = catalog["product_text"].astype(str).tolist()
    print(f"[index] catalog products: {len(product_ids):,}")

    # Single source of truth for row order across all three legs.
    with open(os.path.join(index_dir, "product_ids.json"), "w") as f:
        json.dump(product_ids, f)

    # -- Dense (FAISS) -------------------------------------------------------
    print(f"[index] loading bi-encoder: {biencoder_path}")
    model = SentenceTransformer(biencoder_path)
    faiss_index, _ = build_dense_index(model, product_ids, texts)
    faiss.write_index(faiss_index, os.path.join(index_dir, "dense.faiss"))
    print("[index] dense FAISS index written")

    # -- BM25 ----------------------------------------------------------------
    bm25 = BM25Index().build(product_ids, texts)
    bm25.save(os.path.join(index_dir, "bm25.pkl"))
    print("[index] BM25 index written")

    # -- SPLADE (CSR) --------------------------------------------------------
    splade = SpladeEncoder(model_name=splade_model, max_length=splade_maxlen)
    splade.build_index(product_ids, texts)
    splade.save(os.path.join(index_dir, "splade.npz"))
    print("[index] SPLADE CSR index written")

    paths = {
        "index_dir": index_dir,
        "dense": os.path.join(index_dir, "dense.faiss"),
        "bm25": os.path.join(index_dir, "bm25.pkl"),
        "splade": os.path.join(index_dir, "splade.npz"),
        "product_ids": os.path.join(index_dir, "product_ids.json"),
        "num_products": len(product_ids),
    }
    print(f"[index] done: {paths}")
    return paths
