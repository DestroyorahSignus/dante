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
