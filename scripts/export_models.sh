#!/usr/bin/env bash
set -euo pipefail

echo "TODO: export trained DANTE models for inference — see DANTE_BUILD_PLAN.md §8"

# Export the trained bi-encoder
# cp -r models/biencoder_final  export/biencoder

# Export SPLADE checkpoint
# cp -r models/splade_final     export/splade

# Export ColBERT reranker
# cp -r models/colbert_final    export/colbert

# Export FAISS + BM25 + SPLADE indices
# cp models/faiss.index  models/bm25.pkl  models/splade_index.pkl  export/
