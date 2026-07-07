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

## v0.2 — hard-negative fine-tuning

v0.2 does exactly what the v0.1 "Read" note pointed at: it fine-tunes the dense
bi-encoder on **mined hard negatives** to lift the weakest leg. The winning recipe is
**`gte-modernbert-base` + hard negatives** (v0.1's MNRL recipe, bs=128, ~5151 steps,
plus the hard-negative pairs). The dense leg jumps **R@200 0.627 → 0.698 (+11%)** and
**nDCG@10 0.313 → 0.410**, and the best overall config (**Dense+SPLADE**) reaches
**R@200 0.7296 / nDCG@10 0.4461**.

v0.1 was evaluated on a 2,000-query test sample, v0.2 on an 800-query sample. The two
are comparable: the model-independent rows match across samples — BM25 R@200 0.530→0.517,
SPLADE R@200 0.674→0.677.

| Configuration | R@200 (v0.1) | R@200 (v0.2 gte) | nDCG@10 (v0.1) | nDCG@10 (v0.2 gte) |
|---|---|---|---|---|
| BM25 | 0.530 | 0.517 | 0.321 | 0.317 |
| Dense | 0.627 | **0.698** | 0.313 | **0.410** |
| SPLADE | 0.674 | 0.677 | 0.434 | 0.432 |
| Dense + BM25 (RRF) | 0.678 | 0.692 | 0.363 | 0.393 |
| Dense + SPLADE (RRF) | 0.730 | **0.730** | 0.418 | **0.446** |
| Dense + BM25 + SPLADE (RRF) | 0.719 | 0.715 | 0.424 | 0.428 |
| ↑ + ColBERT rerank | 0.719 | 0.715 | **0.448** | 0.446 |
| ↑ + CE rerank (gte-modernbert) | — | 0.715 | — | 0.330 |
| Dense + SPLADE (RRF k=30) | — | 0.729 | — | 0.448 |
| Dense + BM25 + SPLADE (weighted RRF) | — | 0.728 | — | 0.445 |

**Read:** the fine-tune lifts the dense leg by a wide margin (+11% R@200, +0.097 nDCG@10)
and — crucially — a *stronger dense leg does not fold into a better fused number for free*:
Dense+SPLADE R@200 is flat (0.730 → 0.730) because SPLADE already covered most of what the
dense leg was missing, but the *ranking* quality of that same config improves markedly
(nDCG@10 0.418 → 0.446). The cross-encoder rerank row (nDCG@10 0.330) *underperforms* the
fusion it reranks — a pretrained-not-fine-tuned CE is the wrong tool at this recall depth —
so ColBERT stays the reranker of record. Full numbers in
[`ablation_gte.json`](ablation_gte.json).

### Negative result: the ModernBERT control regressed

The sweep ran a control — **plain `ModernBERT-base` + the same hard negatives** — and it
**regressed the dense leg to R@200 0.5478**, below v0.1's in-batch 0.627. The lesson is the
whole point of the release: **hard negatives only help when the backbone is already
retrieval-pretrained.** ESCI's labels are incomplete, so hard-negative mining inevitably
scoops up *unlabeled relevants* (false negatives) and trains the model to push them away.
A retrieval-pretrained backbone (`gte-modernbert-base`) has enough prior structure to
survive that noise and still net a gain; a raw-MLM backbone (`ModernBERT-base`) gets
poisoned by it and degrades. v0.2 therefore ships the gte base and keeps the ModernBERT-HN
run in the repo as the documented control.

### Deferred / dropped from the sweep

- **`bge-reranker-v2-gemma` (2B)** — too slow to rerank at eval scale; dropped.
- **`bge-reranker-v2-m3`** — the `rerankers` library hangs on load for this checkpoint; dropped.
- **Big-batch CachedMNRL (bs=2048)** — undertrained at 216 steps (one pass at the large
  batch is too few optimizer updates); needs epoch/LR retuning before it's a fair comparison,
  so it's deferred rather than reported. The shipped runs use the bs=128 / ~5151-step recipe.

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
