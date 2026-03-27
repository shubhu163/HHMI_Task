#!/usr/bin/env bash
set -euo pipefail


python ./scripts/visualize_retrieval.py \
  --subset-root ./data/subset \
  --retrieval-json ./data/results/object_retrieval_jrc_hela-2_z00967_r0.json \
  --output-dir ./data/results/figures/object_retrieval_jrc_hela-2_z00967_r0 \
  --top-k 10 \
  --bottom-k 14 \
  --margin 24 \
  --font-size 18 \
  --max-cols 2

python ./scripts/visualize_retrieval.py \
  --subset-root ./data/subset \
  --retrieval-json ./data/results/object_retrieval_jrc_hela-3_z00128_r0.json \
  --output-dir ./data/results/figures/object_retrieval_jrc_hela-3_z00128_r0 \
  --top-k 10 \
  --bottom-k 14 \
  --margin 24 \
  --font-size 18 \
  --max-cols 2
