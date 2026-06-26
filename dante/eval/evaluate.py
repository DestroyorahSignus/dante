"""End-to-end evaluation pipeline — DANTE_BUILD_PLAN.md §5."""


def evaluate(engine, qrels, config):
    """Run the full ablation eval and produce the §5.2 results table.

    Evaluates each configuration (BM25 only, Dense only, SPLADE only, the RRF
    combinations, and + ColBERT rerank) on the ESCI test split, reporting MRR@10,
    nDCG@10 and Recall@{10,50,100,200}. Also produces the per-label breakdown (§5.3).

    Args:
        engine: A DanteSearchEngine (or its component retrievers).
        qrels: Graded relevance judgements {query_id: {product_id: grade}}.
        config: Eval config (k values, labels, output path).

    Returns:
        A dict of per-configuration metric rows.
    """
    raise NotImplementedError("TODO: see DANTE_BUILD_PLAN.md §5")
