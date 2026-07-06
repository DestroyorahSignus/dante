"""End-to-end evaluation / ablation — DANTE_BUILD_PLAN.md §5.

One shared ``evaluate_ranker`` drives EVERY ablation row (and could drive the
trainer's in-loop evaluator), so rows are always comparable — same metric code, same
qrels. ``run_all_ablations`` runs the 10 configs through it and prints the §5.2 table.

Conventions (§3.4 / §5):
  * qrels grades: Exact=3, Substitute=2, Complement=1, Irrelevant=0.
  * recall positives = grade >= 2 (Exact + Substitute).
  * nDCG relevance_map = the full graded dict (uses all 4 grades).
"""
from __future__ import annotations

import json

from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank

POS_GRADE = 2  # Exact + Substitute count as relevant for recall


def evaluate_ranker(rank_fn, queries: dict, qrels: dict, ks=(10, 50, 100, 200)) -> dict:
    """Average MRR@10 / nDCG@10 / Recall@k over the eval queries.

    Args:
        rank_fn: ``query_text -> list[product_id]`` (ranked). ONE interface; every
            retriever/fusion/reranked config implements it.
        queries: ``{query_id: query_text}`` (the eval subset).
        qrels: ``{query_id: {product_id: grade}}`` with Exact=3..Irrelevant=0.
        ks: Recall cutoffs.

    Returns:
        ``{"mrr@10":.., "ndcg@10":.., "recall@<k>":..}`` averaged over queries that
        have at least one positive (grade>=2) judgement.
    """
    ks = tuple(ks)
    sums = {"mrr@10": 0.0, "ndcg@10": 0.0}
    for k in ks:
        sums[f"recall@{k}"] = 0.0
    n = 0

    for qid, qtext in queries.items():
        rel_map = qrels.get(qid, {})
        positives = {pid for pid, g in rel_map.items() if g >= POS_GRADE}
        if not positives:
            continue  # no relevant doc → undefined recall/MRR; skip (keeps rows comparable)
        ranked = rank_fn(qtext)
        sums["mrr@10"] += reciprocal_rank(ranked[:10], positives)
        sums["ndcg@10"] += ndcg_at_k(ranked, rel_map, 10)
        for k in ks:
            sums[f"recall@{k}"] += recall_at_k(ranked, positives, k)
        n += 1

    if n == 0:
        return {m: 0.0 for m in sums}
    return {m: v / n for m, v in sums.items()}


def _subsample_queries(queries: dict, qrels: dict, max_queries: int, seed: int) -> dict:
    """Deterministically subsample to queries that have >=1 positive judgement."""
    import random

    eligible = [
        qid for qid in queries
        if any(g >= POS_GRADE for g in qrels.get(qid, {}).values())
    ]
    eligible.sort()  # stable base order before seeded shuffle
    if max_queries and len(eligible) > max_queries:
        rng = random.Random(seed)
        eligible = rng.sample(eligible, max_queries)
    return {qid: queries[qid] for qid in eligible}


def _format_table(results: dict, ks=(10, 50, 100, 200)) -> str:
    """Render the §5.2 ablation table as fixed-width text."""
    rcols = [f"recall@{k}" for k in ks]
    headers = ["Configuration", "MRR@10", "nDCG@10"] + [f"R@{k}" for k in ks]
    rows = [headers]
    for name, m in results.items():
        rows.append([
            name, f"{m.get('mrr@10', 0):.4f}", f"{m.get('ndcg@10', 0):.4f}",
            *[f"{m.get(c, 0):.4f}" for c in rcols],
        ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    out = []
    for ri, row in enumerate(rows):
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if ri == 0:
            out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(out)


def run_all_ablations(engine, queries: dict, qrels: dict,
                      ks=(10, 50, 100, 200), max_queries: int = 2000,
                      seed: int = 42) -> dict:
    """Run the 10 ablation configs through the shared ``evaluate_ranker`` (§5.2).

    Configs: BM25, Dense, SPLADE, Dense+BM25, Dense+SPLADE, Dense+BM25+SPLADE,
    Dense+BM25+SPLADE + ColBERT rerank, plus three v0.2 rows motivated by the
    v0.1 measurements (Dense+SPLADE beat all-3 → BM25 adds noise; k=10-30 beat
    k=60 on nDCG@10): "+ CE rerank (bge-reranker-base)" (a cross-encoder over the
    SAME fused candidates as the ColBERT row), "Dense+SPLADE (RRF k=30)", and
    "Dense+BM25+SPLADE (weighted RRF)" with BM25 down-weighted to 0.5. All
    retrieval reuses ``engine``'s per-leg / fused helpers, so every row shares
    production's retrieval code.

    Args:
        engine: A constructed ``DanteSearchEngine``.
        queries: ``{query_id: query_text}`` (full test set; subsampled inside).
        qrels: graded judgements.
        ks: recall cutoffs.
        max_queries: subsample cap (so ColBERT stays in budget); ~2000 by default.
        seed: subsample seed.

    Returns:
        ``{"results": {config: {metric: val}}, "table": str, "n_queries": int}``.
    """
    from ..models.colbert_reranker import colbert_rerank, rerank
    from ..models.fusion import reciprocal_rank_fusion

    eval_q = _subsample_queries(queries, qrels, max_queries, seed)
    print(f"[ablation] evaluating {len(eval_q)} queries (of {len(queries)} test queries)")

    def _colbert_rank(qtext):
        fused_ids = engine.fused(qtext, legs=("dense", "bm25", "splade"))
        candidates = [engine.product_db[p] for p in fused_ids if p in engine.product_db]
        reranked = colbert_rerank(qtext, candidates, top_k=max(ks),
                                  model=engine.colbert_model)
        return [c["product_id"] for c in reranked]

    def _ce_rank(qtext):
        # SAME fused candidates as the ColBERT row — only the reranker differs, so
        # the two rows isolate late-interaction vs cross-encoder reranking.
        fused_ids = engine.fused(qtext, legs=("dense", "bm25", "splade"))
        candidates = [engine.product_db[p] for p in fused_ids if p in engine.product_db]
        reranked = rerank(qtext, candidates, top_k=max(ks),
                          model_name="BAAI/bge-reranker-base",
                          model_type="cross-encoder")
        return [c["product_id"] for c in reranked]

    def _fused_custom(qtext, legs, k=None, weights=None):
        """Like ``engine.fused`` but with a per-config RRF k and/or leg weights.

        Uses the SAME per-leg retrieval helpers (and leg_top_k depth) as
        ``engine.fused``, so these rows differ from the standard fusion rows
        ONLY in the fusion constant/weights — a clean ablation.
        """
        leg_fns = {"dense": engine._dense, "bm25": engine._bm25, "splade": engine._splade}
        ranked = [leg_fns[leg](qtext, engine.leg_top_k) for leg in legs]
        fused = reciprocal_rank_fusion(
            ranked, k=engine.rrf_k if k is None else k,
            top_n=engine.top_n, weights=weights,
        )
        return [pid for pid, _ in fused]

    configs = {
        "BM25":                  lambda q: engine.bm25_only(q),
        "Dense":                 lambda q: engine.dense_only(q),
        "SPLADE":                lambda q: engine.splade_only(q),
        "Dense+BM25":            lambda q: engine.fused(q, legs=("dense", "bm25")),
        "Dense+SPLADE":          lambda q: engine.fused(q, legs=("dense", "splade")),
        "Dense+BM25+SPLADE":     lambda q: engine.fused(q, legs=("dense", "bm25", "splade")),
        "+ ColBERT rerank":      _colbert_rank,
        # --- v0.2 rows (v0.1-measured: BM25 noisy, k=10-30 > k=60) --------------
        # Cross-encoder rerank of the SAME fused top-200 as the ColBERT row.
        "+ CE rerank (bge-reranker-base)": _ce_rank,
        # Best v0.1 pair, re-fused at the sweep's sweet-spot constant.
        "Dense+SPLADE (RRF k=30)":
            lambda q: _fused_custom(q, ("dense", "splade"), k=30),
        # Keep BM25's lexical signal but halve its vote instead of dropping it.
        # weights are parallel to the legs tuple: dense=1.0, bm25=0.5, splade=1.0.
        "Dense+BM25+SPLADE (weighted RRF)":
            lambda q: _fused_custom(q, ("dense", "bm25", "splade"),
                                    weights=[1.0, 0.5, 1.0]),
    }

    results = {}
    for name, rank_fn in configs.items():
        print(f"[ablation] running config: {name}")
        results[name] = evaluate_ranker(rank_fn, eval_q, qrels, ks=ks)

    table = _format_table(results, ks=ks)
    print("\n" + table + "\n")
    return {"results": results, "table": table, "n_queries": len(eval_q)}


# ============================================================================
# EVAL-ENRICHMENT (no retrain) — DANTE_BUILD_PLAN.md §4.2 / §14 (the D4 GPU pass)
#
#   (a) dim_truncation_ablation — robustness of the dense leg to embedding
#       truncation. The bi-encoder is plain MNRL (NOT MatryoshkaLoss), so this is a
#       "what does naive truncation cost?" ablation, not a Matryoshka claim: slice the
#       768-d catalog embeddings to the first N dims, L2-RENORMALIZE, build a fresh
#       IndexFlatIP, and measure dense recall@K + nDCG@10. The cheap-serving story.
#   (b) rrf_k_sweep — re-fuse the SAME three legs' top-1000 lists at several RRF k
#       values and report nDCG@10 + R@200 to justify k=60.
#
# Both reuse the shared evaluate_ranker + the SAME subsampled eval queries/seed as
# run_all_ablations, so the rows are directly comparable to the baseline ablation.
# ============================================================================

def _format_dim_table(results: dict, ks=(10, 50, 100, 200)) -> str:
    """Render the dim-truncation ablation as fixed-width text."""
    rcols = [f"recall@{k}" for k in ks]
    headers = ["Dense dim", "nDCG@10"] + [f"R@{k}" for k in ks]
    rows = [headers]
    for dim, m in results.items():
        rows.append([
            str(dim), f"{m.get('ndcg@10', 0):.4f}",
            *[f"{m.get(c, 0):.4f}" for c in rcols],
        ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    out = []
    for ri, row in enumerate(rows):
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if ri == 0:
            out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(out)


def _format_rrf_table(results: dict) -> str:
    """Render the RRF-k sweep as fixed-width text."""
    headers = ["RRF k", "nDCG@10", "R@200"]
    rows = [headers]
    for k, m in results.items():
        rows.append([str(k), f"{m.get('ndcg@10', 0):.4f}", f"{m.get('recall@200', 0):.4f}"])
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    out = []
    for ri, row in enumerate(rows):
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if ri == 0:
            out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(out)


def dim_truncation_ablation(model, catalog_ids, catalog_texts, queries, qrels,
                            dims=(768, 256, 128), ks=(10, 50, 100, 200),
                            max_queries: int = 2000, seed: int = 42,
                            encode_batch_size: int = 256, leg_top_k: int = 1000) -> dict:
    """Dense-leg dimension-truncation robustness ablation (NO retrain).

    Encodes the catalog AND the eval queries ONCE at full width, then for each target
    ``dim`` slices the embeddings to the first ``dim`` columns, L2-renormalizes, builds
    a fresh ``faiss.IndexFlatIP``, and scores dense-only retrieval through the shared
    ``evaluate_ranker``. Truncate-THEN-renormalize is the documented cheap-index recipe
    (§4.2 / §13 R5). The same subsampled eval queries (seed) as ``run_all_ablations``
    are used, so the 768-d row reproduces the baseline "Dense" ablation row.

    Args:
        model: A loaded ``SentenceTransformer`` (the trained bi-encoder).
        catalog_ids: Parallel product ids (full retrieval pool).
        catalog_texts: Parallel product texts.
        queries: ``{query_id: query_text}`` (full test set; subsampled inside).
        qrels: graded ``{query_id: {product_id: grade}}``.
        dims: Truncation widths to test (must be <= the model's full dim).
        ks: Recall cutoffs.
        max_queries: subsample cap (shared with the baseline ablation).
        seed: subsample seed (shared with the baseline ablation).
        encode_batch_size: catalog encode batch size.
        leg_top_k: per-query retrieval depth (matches serving's leg_top_k).

    Returns:
        ``{"results": {dim: {metric: val}}, "table": str, "n_queries": int,
           "full_dim": int}``.
    """
    import faiss
    import numpy as np

    eval_q = _subsample_queries(queries, qrels, max_queries, seed)
    qids = list(eval_q)
    print(f"[dim-ablation] evaluating {len(eval_q)} queries; dims={list(dims)}")

    # Encode the catalog ONCE at full width (un-normalized — we renormalize per dim).
    print(f"[dim-ablation] encoding catalog ({len(catalog_ids):,} products) once @ full dim ...")
    cat_emb = model.encode(
        list(catalog_texts), batch_size=encode_batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype("float32")
    full_dim = int(cat_emb.shape[1])

    # Encode the eval queries ONCE at full width too.
    q_emb_full = model.encode(
        [eval_q[qid] for qid in qids], batch_size=encode_batch_size,
        show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False,
    ).astype("float32")

    k_search = min(leg_top_k, len(catalog_ids))
    results: dict = {}
    for dim in dims:
        if dim > full_dim:
            print(f"[dim-ablation] skip dim={dim} > model dim {full_dim}")
            continue
        # Slice to the first `dim` columns, then RENORMALIZE (truncate-then-normalize).
        cat_d = np.ascontiguousarray(cat_emb[:, :dim]).astype("float32")
        faiss.normalize_L2(cat_d)
        index = faiss.IndexFlatIP(dim)
        index.add(cat_d)

        q_d = np.ascontiguousarray(q_emb_full[:, :dim]).astype("float32")
        faiss.normalize_L2(q_d)
        scores, idx = index.search(q_d, k_search)

        # Precompute the per-query ranked id list (one batched FAISS search above),
        # then route through the shared evaluate_ranker for identical metric code.
        # Per-qid ranked lists from the one batched FAISS search above. We key by qid
        # (not query text) to stay exact when two queries share text, then route
        # through _evaluate_precomputed — the SAME metric code as evaluate_ranker.
        ranked_by_qid = {
            qids[row]: [catalog_ids[i] for i in idx[row] if i >= 0]
            for row in range(len(qids))
        }
        metrics = _evaluate_precomputed(ranked_by_qid, eval_q, qrels, ks=ks)
        results[dim] = metrics
        print(f"[dim-ablation] dim={dim}: "
              f"nDCG@10={metrics['ndcg@10']:.4f} R@200={metrics.get('recall@200', 0):.4f}")
        del cat_d, index

    table = _format_dim_table(results, ks=ks)
    print("\n[dim-ablation]\n" + table + "\n")
    return {"results": results, "table": table, "n_queries": len(eval_q),
            "full_dim": full_dim}


def _evaluate_precomputed(ranked_by_qid: dict, queries: dict, qrels: dict,
                          ks=(10, 50, 100, 200)) -> dict:
    """Like ``evaluate_ranker`` but takes already-ranked id lists keyed BY query id.

    Used by the enrichment ablations, which batch their retrieval and so already have
    per-qid ranked lists. Same metric code / same skip-if-no-positive rule as
    ``evaluate_ranker`` → rows stay comparable.
    """
    ks = tuple(ks)
    sums = {"mrr@10": 0.0, "ndcg@10": 0.0}
    for k in ks:
        sums[f"recall@{k}"] = 0.0
    n = 0
    for qid in queries:
        rel_map = qrels.get(qid, {})
        positives = {pid for pid, g in rel_map.items() if g >= POS_GRADE}
        if not positives:
            continue
        ranked = ranked_by_qid.get(qid, [])
        sums["mrr@10"] += reciprocal_rank(ranked[:10], positives)
        sums["ndcg@10"] += ndcg_at_k(ranked, rel_map, 10)
        for k in ks:
            sums[f"recall@{k}"] += recall_at_k(ranked, positives, k)
        n += 1
    if n == 0:
        return {m: 0.0 for m in sums}
    return {m: v / n for m, v in sums.items()}


def rrf_k_sweep(engine, queries, qrels, k_values=(10, 30, 60, 100),
                legs=("dense", "bm25", "splade"), ks=(10, 50, 100, 200),
                max_queries: int = 2000, seed: int = 42) -> dict:
    """RRF constant sweep over the SAME three legs' top-1000 lists (NO retrain).

    For each query the three legs' ranked lists are retrieved ONCE (cached), then
    re-fused at every ``k`` in ``k_values`` via ``reciprocal_rank_fusion`` — so the
    sweep costs one retrieval pass, not one per k. Reports nDCG@10 + R@200 to justify
    the default ``k=60``. Same subsampled eval queries (seed) as ``run_all_ablations``.

    Args:
        engine: A constructed ``DanteSearchEngine`` (provides the per-leg helpers).
        queries: ``{query_id: query_text}`` (full test set; subsampled inside).
        qrels: graded judgements.
        k_values: RRF k constants to sweep.
        legs: which legs to fuse (default all three, matching production).
        ks: recall cutoffs reported.
        max_queries: subsample cap (shared with the baseline ablation).
        seed: subsample seed (shared with the baseline ablation).

    Returns:
        ``{"results": {k: {metric: val}}, "table": str, "n_queries": int}``.
    """
    from ..models.fusion import reciprocal_rank_fusion

    eval_q = _subsample_queries(queries, qrels, max_queries, seed)
    print(f"[rrf-sweep] evaluating {len(eval_q)} queries; k_values={list(k_values)}")

    leg_fns = {"dense": engine._dense, "bm25": engine._bm25, "splade": engine._splade}
    top_n = max(ks)  # fuse deep enough for R@200

    # Retrieve each leg ONCE per query and cache the ranked lists.
    cached: dict = {}
    for qid, qtext in eval_q.items():
        cached[qid] = [leg_fns[leg](qtext, engine.leg_top_k) for leg in legs]

    results: dict = {}
    for k in k_values:
        ranked_by_qid = {
            qid: [pid for pid, _ in reciprocal_rank_fusion(lists, k=k, top_n=top_n)]
            for qid, lists in cached.items()
        }
        metrics = _evaluate_precomputed(ranked_by_qid, eval_q, qrels, ks=ks)
        results[k] = metrics
        print(f"[rrf-sweep] k={k}: "
              f"nDCG@10={metrics['ndcg@10']:.4f} R@200={metrics.get('recall@200', 0):.4f}")

    table = _format_rrf_table(results)
    print("\n[rrf-sweep]\n" + table + "\n")
    return {"results": results, "table": table, "n_queries": len(eval_q)}


def evaluate(engine, qrels, config):
    """Thin caller that loads queries from config and runs the full ablation (§5.2)."""
    serving = config.get("serving", {}) if isinstance(config, dict) else {}
    eval_cfg = config.get("eval", {}) if isinstance(config, dict) else {}
    queries_path = serving.get("queries_path",
                               eval_cfg.get("queries_path", "/artifacts/data/queries.json"))
    with open(queries_path) as f:
        queries = json.load(f)
    return run_all_ablations(
        engine, queries, qrels,
        ks=tuple(eval_cfg.get("ks", (10, 50, 100, 200))),
        max_queries=eval_cfg.get("max_queries", 2000),
        seed=eval_cfg.get("seed", 42),
    )
