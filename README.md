# DANTE — Multi-Stage Hybrid Product Search Engine

DANTE is a **multi-stage hybrid product search engine** built on the Amazon ESCI
(Shopping Queries) dataset. It combines a **fine-tuned ModernBERT bi-encoder** (dense
retrieval), **SPLADE-v3 learned-sparse retrieval**, **BM25 lexical retrieval**, fused
with **Reciprocal Rank Fusion (RRF)**, and reranked by a **ColBERT late-interaction
reranker** (answerai-colbert-small-v1). The system includes hard-negative mining,
graded relevance evaluation (E/S/C/I labels → MRR, nDCG@10, Recall@K), and a
Gradio demo UI. Every component runs inference on a Kaggle T4 16GB GPU.

## Architecture

```
Query → ┌─ Dense (ModernBERT bi-encoder) → top-1000 ─┐
        ├─ Sparse (SPLADE-v3)            → top-1000 ─┤→ RRF Fusion → top-200 → ColBERT Rerank → top-20
        └─ Lexical (BM25)               → top-1000 ─┘
```

All three retrieval legs run in parallel. RRF fuses them into a single ranked list.
ColBERT reranks the top-200 for the final ordering.

## Results (ablation)

Amazon ESCI (US, reduced), 2,000 held-out test queries (query-disjoint split). Fine-tuned
ModernBERT bi-encoder (3 epochs, MNRL in-batch negatives); SPLADE + ColBERT pretrained.
Retriever ceiling (canonical-query recall@1 on the dense index) = **0.993**.

| Configuration | MRR@10 | nDCG@10 | R@10 | R@100 | R@200 |
|---|---|---|---|---|---|
| BM25 only | 0.542 | 0.321 | 0.192 | 0.457 | 0.530 |
| Dense only (ModernBERT) | 0.531 | 0.313 | 0.196 | 0.527 | 0.627 |
| SPLADE only | 0.659 | 0.434 | 0.267 | 0.594 | 0.674 |
| Dense + BM25 (RRF) | 0.596 | 0.363 | 0.227 | 0.584 | 0.678 |
| Dense + SPLADE (RRF) | 0.655 | 0.418 | 0.262 | 0.646 | **0.730** |
| Dense + BM25 + SPLADE (RRF) | 0.660 | 0.424 | 0.265 | 0.637 | 0.719 |
| **↑ + ColBERT rerank** | **0.679** | **0.448** | **0.276** | 0.630 | 0.719 |

**Read:** fusion lifts recall (best R@200 = 0.730, Dense+SPLADE, vs 0.674 for the best single
leg) and ColBERT reranking lifts ranking quality (+0.034 MRR@10 / +0.024 nDCG@10 over the fused
top-200, recall unchanged since it only reorders). SPLADE is the strongest single signal; the
in-batch-trained dense leg is the weakest — hard-negative mining (next iteration) is the lever to
lift it. Full numbers in [`ablation_results.json`](ablation_results.json).

## Install

```
pip install git+https://github.com/DestroyorahSignus/dante.git
```

## How to run

TODO: end-to-end reproduction in one command (see `scripts/train_all.sh`).

```
bash scripts/train_all.sh
```

## Train the bi-encoder on Modal (A100)

`modal_train.py` is a self-contained Modal job that fine-tunes the ModernBERT
bi-encoder on ESCI. It is fully isolated — its own app (`dante-train`) and volume
(`dante-artifacts`), **no secrets attached**, no database, no shared state with any
other project on your Modal workspace. (SPLADE and ColBERT use pretrained checkpoints,
so the bi-encoder is the only thing that needs an A100.)

```
pip install modal && modal token new          # one-time auth

modal run modal_train.py --stage data                          # build train/eval data only
modal run modal_train.py --stage all --limit 5000 --epochs 1   # quick smoke test
modal run modal_train.py --stage all                            # full run (~1-1.5h A100)
```

The `data` stage produces **leakage-free** artifacts on the volume: `train/` + `val/`
(MNRL pairs, **split by query** so no query is in both), plus the eval set the ablation
needs — `catalog.parquet` (full retrieval pool), `qrels.json` (graded Exact=3…Irrelevant=0)
and `queries.json`. Train/test are query-disjoint (asserted), and positives are deduped +
capped per query.

Pull the trained model to your machine (then push to HF Hub / Kaggle — not git):

```
modal volume get dante-artifacts /biencoder_final ./models/dante_biencoder
```

Clean up when done (only touches this volume):

```
modal volume delete dante-artifacts
```
