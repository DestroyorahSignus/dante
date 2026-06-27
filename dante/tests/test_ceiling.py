"""Light CPU SPLADE-expansion sanity (§4.3).

The HEAVY GPU retriever-ceiling check (each product's own text → self in top-1/10
against the full dense index) lives in the Modal ``preflight`` stage. Here we only do
a cheap CPU sanity: if the pretrained SPLADE model loads, a query must expand into a
non-empty set of sensible term weights. Skips if the model can't be loaded.
"""
import pytest

pytest.importorskip("transformers")
pytest.importorskip("scipy")
pytest.importorskip("torch")

SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


@pytest.fixture(scope="module")
def encoder():
    from dante.models.splade import SpladeEncoder

    try:
        return SpladeEncoder(model_name=SPLADE_MODEL, device="cpu")
    except Exception as exc:  # download blocked / OOM on a tiny box
        pytest.skip(f"could not load {SPLADE_MODEL}: {exc}")


def test_expansion_is_non_empty(encoder):
    from dante.models.splade import visualize_expansion

    terms = visualize_expansion("running shoes", encoder, top_k_terms=20)
    assert len(terms) > 0
    # weights are positive and sorted descending.
    weights = [w for _, w in terms]
    assert all(w > 0 for w in weights)
    assert weights == sorted(weights, reverse=True)


def test_csr_search_self_hit(encoder):
    """Build a tiny CSR index and confirm a doc retrieves itself at rank 1."""
    ids = ["a", "b", "c"]
    docs = [
        "wireless bluetooth headphones",
        "stainless steel water bottle",
        "mechanical gaming keyboard",
    ]
    encoder.build_index(ids, docs, batch_log=0)
    hits = encoder.search(docs[0], top_k=1)
    assert hits[0][0] == "a"
