"""Gradio demo UI — DANTE_BUILD_PLAN.md §6."""


def create_demo(engine):
    """Build the DANTE Gradio search demo.

    A search box + top-k slider returns HTML result cards (title, brand, relevance
    score, which retrievers contributed: BM25 / Dense / SPLADE) plus a SPLADE term
    expansion panel for query interpretability.

    Args:
        engine: A constructed DanteSearchEngine.

    Returns:
        A gradio.Blocks demo (call ``.launch()`` to serve).
    """
    raise NotImplementedError("TODO: see DANTE_BUILD_PLAN.md §6")
