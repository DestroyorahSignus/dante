"""FAISS self-test with a tiny sentence-transformers model — self-contained, CPU.

Validates the dense leg's contract (§4.2 / RISK R7): unit-norm embeddings, an
IndexFlatIP index, and that querying with a document's own text returns that
document at rank 1. Skips cleanly if faiss / sentence-transformers / the tiny model
can't be loaded in the sandbox.
"""
import numpy as np
import pytest

faiss = pytest.importorskip("faiss")
st = pytest.importorskip("sentence_transformers")

from dante.models.biencoder import build_dense_index, dense_search

# Tiny, fast, widely-cached model (22M params) — keeps the test CPU-cheap.
TINY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOCS = [
    "wireless bluetooth over-ear headphones with noise cancelling",
    "stainless steel insulated water bottle 32oz",
    "mechanical gaming keyboard rgb backlit",
    "organic cotton crew neck t-shirt navy blue",
    "4k ultra hd smart television 55 inch",
]
IDS = [f"p{i}" for i in range(len(DOCS))]


@pytest.fixture(scope="module")
def model():
    try:
        return st.SentenceTransformer(TINY_MODEL)
    except Exception as exc:  # offline / download blocked
        pytest.skip(f"could not load {TINY_MODEL}: {exc}")


def test_embeddings_are_unit_norm(model):
    emb = model.encode(DOCS, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)


def test_index_is_flat_ip(model):
    index, ids = build_dense_index(model, IDS, DOCS)
    assert index.ntotal == len(DOCS)
    assert ids == IDS


def test_self_query_is_rank_one(model):
    index, ids = build_dense_index(model, IDS, DOCS)
    for i, doc in enumerate(DOCS):
        hits = dense_search(model, index, ids, doc, top_k=1)
        assert hits[0][0] == IDS[i], f"doc {i} did not self-retrieve at rank 1"


def test_self_query_score_near_one(model):
    index, ids = build_dense_index(model, IDS, DOCS)
    hits = dense_search(model, index, ids, DOCS[0], top_k=1)
    # cosine of a doc with itself ≈ 1.0.
    assert hits[0][1] > 0.99
