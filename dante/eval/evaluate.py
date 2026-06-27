"""End-to-end evaluation / ablation — DANTE_BUILD_PLAN.md §5.

One shared ``evaluate_ranker`` drives EVERY ablation row (and could drive the
trainer's in-loop evaluator), so rows are always comparable — same metric code, same
qrels. ``run_all_ablations`` runs the 7 configs through it and prints the §5.2 table.

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
    """Run the 7 ablation configs through the shared ``evaluate_ranker`` (§5.2).

    Configs: BM25, Dense, SPLADE, Dense+BM25, Dense+SPLADE, Dense+BM25+SPLADE, and
    Dense+BM25+SPLADE + ColBERT rerank. All retrieval reuses ``engine``'s per-leg /
    fused helpers, so every row shares production's retrieval code.

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
    from ..models.colbert_reranker import colbert_rerank

    eval_q = _subsample_queries(queries, qrels, max_queries, seed)
    print(f"[ablation] evaluating {len(eval_q)} queries (of {len(queries)} test queries)")

    def _colbert_rank(qtext):
        fused_ids = engine.fused(qtext, legs=("dense", "bm25", "splade"))
        candidates = [engine.product_db[p] for p in fused_ids if p in engine.product_db]
        reranked = colbert_rerank(qtext, candidates, top_k=max(ks),
                                  model=engine.colbert_model)
        return [c["product_id"] for c in reranked]

    configs = {
        "BM25":                  lambda q: engine.bm25_only(q),
        "Dense":                 lambda q: engine.dense_only(q),
        "SPLADE":                lambda q: engine.splade_only(q),
        "Dense+BM25":            lambda q: engine.fused(q, legs=("dense", "bm25")),
        "Dense+SPLADE":          lambda q: engine.fused(q, legs=("dense", "splade")),
        "Dense+BM25+SPLADE":     lambda q: engine.fused(q, legs=("dense", "bm25", "splade")),
        "+ ColBERT rerank":      _colbert_rank,
    }

    results = {}
    for name, rank_fn in configs.items():
        print(f"[ablation] running config: {name}")
        results[name] = evaluate_ranker(rank_fn, eval_q, qrels, ks=ks)

    table = _format_table(results, ks=ks)
    print("\n" + table + "\n")
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
