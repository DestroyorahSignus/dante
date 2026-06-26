"""Build FAISS + inverted indices — DANTE_BUILD_PLAN.md §4.6 / §8."""


def build_indices(config):
    """Build all serving indices: FAISS dense, BM25 lexical, SPLADE sparse.

    Encodes the catalog with each retriever and persists the FAISS index, the BM25
    pickle, and the SPLADE inverted/sparse index to the paths in config.

    Args:
        config: Paths and build params for each index.

    Returns:
        A dict of built-index handles / paths.
    """
    raise NotImplementedError("TODO: see DANTE_BUILD_PLAN.md §4.6")
