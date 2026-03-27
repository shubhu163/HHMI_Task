#!/usr/bin/env bash
set -euo pipefail

python ./scripts/retrieve_mito_objects.py \
  --subset-root ./data/subset \
  --dense-root ./data/dense_embeddings \
  --datasets jrc_hela-2 jrc_hela-3 jrc_jurkat-1 \
  --query-dataset jrc_hela-2 \
  --query-slice z00967 \
  --query-rank 0 \
  --min-object-pixels 80 \
  --top-k 10 \
  --output-json ./data/results/object_retrieval_jrc_hela-2_z00967_r0.json

python ./scripts/retrieve_mito_objects.py \
  --subset-root ./data/subset \
  --dense-root ./data/dense_embeddings \
  --datasets jrc_hela-2 jrc_hela-3 jrc_jurkat-1 \
  --query-dataset jrc_hela-3 \
  --query-slice z00128 \
  --query-rank 0 \
  --min-object-pixels 80 \
  --top-k 10 \
  --output-json ./data/results/object_retrieval_jrc_hela-3_z00128_r0.json
