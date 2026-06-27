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

| Configuration | MRR@10 | nDCG@10 | R@10 | R@100 | R@200 |
|---|---|---|---|---|---|
| BM25 only | | | | | |
| Dense only (ModernBERT) | | | | | |
| SPLADE only | | | | | |
| Dense + BM25 (RRF) | | | | | |
| Dense + SPLADE (RRF) | | | | | |
| Dense + BM25 + SPLADE (RRF) | | | | | |
| ↑ + ColBERT rerank | | | | | |

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
