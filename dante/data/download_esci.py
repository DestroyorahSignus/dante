"""Download + preprocess the Amazon ESCI dataset — DANTE_BUILD_PLAN.md §3.2."""


def download_esci(output_dir: str = "data/raw"):
    """Download and preprocess the Amazon ESCI dataset.

    Loads ``tasksource/esci`` from HuggingFace (the reduced ~1.1M-pair variant),
    filters to English (``product_locale == 'us'``), builds the combined
    ``product_text`` field, splits 80/10/10 by QUERY (same query stays in one split),
    and saves queries.parquet, products.parquet, qrels.parquet to output_dir.

    Args:
        output_dir: Directory to write the preprocessed parquet files.
    """
    raise NotImplementedError("TODO: see DANTE_BUILD_PLAN.md §3.2")
