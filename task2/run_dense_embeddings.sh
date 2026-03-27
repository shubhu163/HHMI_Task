#!/usr/bin/env bash
set -euo pipefail

python ./scripts/build_dense_embeddings.py \
  --emb-root ./data/embeddings \
  --output-root ./data/dense_embeddings \
  --datasets \
    jrc_hela-2 jrc_hela-3 jrc_jurkat-1 \
  --l2-normalize
