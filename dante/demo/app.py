"""Gradio demo UI — DANTE_BUILD_PLAN.md §6.

A single search box + top-k slider that runs the REAL pipeline
(``DanteSearchEngine.search``) and renders results as HTML cards: product title,
brand, the final (rerank) rank, and which retrieval legs contributed
(BM25 / Dense / SPLADE ticks). A side panel shows SPLADE's learned term expansion
for the query, the project's interpretability hook (§4.3).

``import gradio`` lives INSIDE :func:`create_demo` so this module imports without
gradio installed, and the heavy engine build is confined to ``__main__`` so merely
importing the module never triggers a model load. Launch with::

    python -m dante.demo.app --config configs/default.yaml
"""
from __future__ import annotations

import html
from typing import Any

# Retriever legs we attribute results to, in display order.
_LEGS = ("bm25", "dense", "splade")
_LEG_LABELS = {"bm25": "BM25", "dense": "Dense", "splade": "SPLADE"}


def _split_product_text(product_text: str) -> tuple[str, str]:
    """Pull ``(title, brand)`` out of a ``title [SEP] brand [SEP] ...`` text (§3.3)."""
    parts = [p.strip() for p in str(product_text).split("[SEP]")]
    title = parts[0] if parts and parts[0] else "(untitled product)"
    brand = parts[1] if len(parts) > 1 and parts[1] else ""
    return title, brand


def _leg_contributions(engine: Any, query: str, product_ids: list[str],
                       depth: int = 200) -> dict[str, set[str]]:
    """Map each retrieval leg to the set of (returned) product ids it surfaced.

    Re-runs the per-leg ablation helpers (``bm25_only`` / ``dense_only`` /
    ``splade_only``) to a shallow ``depth`` so each result card can show which
    legs voted for it. Any leg that errors (e.g. SPLADE disabled) is skipped — the
    demo degrades gracefully rather than crashing.
    """
    wanted = set(product_ids)
    contributions: dict[str, set[str]] = {}
    for leg in _LEGS:
        leg_fn = getattr(engine, f"{leg}_only", None)
        if leg_fn is None:
            continue
        try:
            ranked = leg_fn(query, depth)
        except Exception:  # a missing/broken leg must not take the demo down
            continue
        contributions[leg] = {pid for pid in ranked if pid in wanted}
    return contributions


def _ticks_html(product_id: str, contributions: dict[str, set[str]]) -> str:
    """Render the per-leg contribution ticks for one product as inline HTML."""
    chips = []
    for leg in _LEGS:
        if leg not in contributions:
            continue
        hit = product_id in contributions[leg]
        mark = "✓" if hit else "—"  # check vs em-dash
        color = "#2e7d32" if hit else "#b0b0b0"
        chips.append(
            f'<span style="margin-right:10px;color:{color};font-weight:600;">'
            f"{mark} {_LEG_LABELS[leg]}</span>"
        )
    return "".join(chips)


def _results_html(results: list[dict], contributions: dict[str, set[str]]) -> str:
    """Render the ranked result dicts as a stack of HTML cards."""
    if not results:
        return (
            '<div style="padding:1em;color:#666;">'
            "No matching products found. Try a broader query.</div>"
        )

    cards = []
    for rank, item in enumerate(results, start=1):
        product_id = str(item.get("product_id", ""))
        title, brand = _split_product_text(item.get("product_text", ""))
        ticks = _ticks_html(product_id, contributions)
        brand_html = (
            f'<span style="color:#555;">{html.escape(brand)}</span> &middot; '
            if brand else ""
        )
        cards.append(
            '<div style="border:1px solid #e0e0e0;border-radius:8px;padding:12px 14px;'
            'margin-bottom:8px;background:#fff;">'
            f'<div style="font-weight:700;font-size:1.02em;margin-bottom:2px;">'
            f"#{rank}&nbsp;&nbsp;{html.escape(title)}</div>"
            f'<div style="font-size:0.9em;margin-bottom:6px;">{brand_html}'
            f'<span style="color:#999;">id {html.escape(product_id)}</span></div>'
            f'<div style="font-size:0.85em;">{ticks}</div>'
            "</div>"
        )
    return "".join(cards)


def _splade_expansion(engine: Any, query: str, top_k_terms: int = 20) -> dict[str, float]:
    """Return SPLADE's term expansion for ``query`` as an ordered ``{term: weight}``.

    Prefers ``engine.splade.visualize_expansion(query)`` if that method exists,
    else falls back to the module-level :func:`dante.models.splade.visualize_expansion`.
    Returns an empty dict (not an error) if SPLADE is unavailable.
    """
    splade = getattr(engine, "splade", None)
    if splade is None:
        return {}
    try:
        method = getattr(splade, "visualize_expansion", None)
        if callable(method):
            pairs = method(query, top_k_terms=top_k_terms)
        else:
            from ..models.splade import visualize_expansion

            pairs = visualize_expansion(query, splade, top_k_terms=top_k_terms)
        return {str(term): round(float(weight), 4) for term, weight in pairs}
    except Exception:
        return {}


def create_demo(engine: Any):  # noqa: ANN201 — gradio.Blocks type is import-guarded
    """Build the DANTE Gradio search demo.

    A search box + top-k slider returns HTML result cards (title, brand, final rank,
    which retrievers contributed: BM25 / Dense / SPLADE) plus a SPLADE term-expansion
    panel for query interpretability. The Search button runs the real pipeline,
    ``engine.search(query, top_k=k)``.

    Args:
        engine: A constructed :class:`dante.serving.search_engine.DanteSearchEngine`
            (or any object exposing ``.search(query, top_k)`` and, optionally, the
            per-leg ``*_only`` helpers and a ``.splade`` encoder).

    Returns:
        A ``gradio.Blocks`` demo (call ``.launch()`` to serve it).
    """
    import gradio as gr

    def search_fn(query: str, top_k: float) -> tuple[str, dict[str, float]]:
        """Run the pipeline and return ``(results_html, splade_expansion)``."""
        query = (query or "").strip()
        if not query:
            empty = (
                '<div style="padding:1em;color:#666;">'
                "Enter a search query to begin.</div>"
            )
            return empty, {}

        k = max(1, int(top_k))
        try:
            results = engine.search(query, top_k=k)
        except Exception as exc:  # never crash the UI on a pipeline error
            err = (
                '<div style="padding:1em;color:#b00020;">'
                f"Search failed: {html.escape(str(exc))}</div>"
            )
            return err, {}

        product_ids = [str(r.get("product_id", "")) for r in results]
        contributions = _leg_contributions(engine, query, product_ids)
        return _results_html(results, contributions), _splade_expansion(engine, query)

    with gr.Blocks(title="DANTE Search") as demo:
        gr.Markdown(
            "# \U0001f50d DANTE — Multi-Stage Hybrid Product Search\n"
            "Dense (ModernBERT) + SPLADE + BM25 → RRF fusion → ColBERT rerank."
        )
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="Search query",
                    placeholder="wireless bluetooth headphones",
                    autofocus=True,
                )
            with gr.Column(scale=1):
                top_k = gr.Slider(
                    minimum=5, maximum=50, value=10, step=1, label="Results"
                )
        btn = gr.Button("Search", variant="primary")

        with gr.Row():
            with gr.Column(scale=3):
                output = gr.HTML(label="Results")
            with gr.Column(scale=1):
                expansion = gr.JSON(label="SPLADE term expansion")

        btn.click(search_fn, [query, top_k], [output, expansion])
        query.submit(search_fn, [query, top_k], [output, expansion])

    return demo


def _load_config(path: str) -> dict:
    """Load the YAML config that points the engine at its trained indices/models."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Heavy imports (engine -> torch/faiss/sentence-transformers) live here so that
    # merely importing this module never triggers a model load. Run as a module:
    #   python -m dante.demo.app --config configs/default.yaml
    import argparse

    from ..serving.search_engine import DanteSearchEngine

    parser = argparse.ArgumentParser(description="Launch the DANTE Gradio demo.")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to the YAML config (default: configs/default.yaml).")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    config = _load_config(args.config)
    engine = DanteSearchEngine(config)
    demo = create_demo(engine)
    demo.launch(share=args.share, server_name=args.server_name,
                server_port=args.server_port)
