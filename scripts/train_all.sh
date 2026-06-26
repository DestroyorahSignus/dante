#!/usr/bin/env bash
set -euo pipefail

echo "TODO: one-shot DANTE training pipeline — see DANTE_BUILD_PLAN.md §8"

# Step 1: Download ESCI, preprocess, split
# python -m dante.data.download_esci --output_dir data/raw

# Step 2: Build BM25 index
# python -m dante.models.bm25 --build

# Step 3: Mine hard negatives from BM25
# python -m dante.data.hard_negatives

# Step 4: Train ModernBERT bi-encoder
# python -m dante.models.biencoder --train --config configs/default.yaml

# Step 5: Build FAISS dense index from trained bi-encoder
# python -m dante.serving.index_builder --dense

# Step 6: Mine cross-retriever hard negatives (dense+BM25)
# Step 7: Fine-tune SPLADE (or download pretrained)
# Step 8: Build SPLADE sparse index
# Step 9: Fine-tune ColBERT on ESCI (optional)
# Step 10: Run full eval pipeline (ablation table)
# python -m dante.eval.evaluate --config configs/default.yaml
