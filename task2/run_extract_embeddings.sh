#!/usr/bin/env bash
set -euo pipefail

python ./scripts/extract_dino_embeddings.py \
  --subset-root ./data/subset \
  --output-root ./data/embeddings \
  --datasets jrc_hela-2 jrc_hela-3 jrc_jurkat-1 \
  --hub-repo-dir dinov3 \
  --hub-entrypoint dinov3_vits16plus \
  --weights weights/dinov3_vits16_plus_pretrain_lvd1689m.pth \
  --batch-size 8 \
  --tile-size 384 \
  --tile-stride 192 \
  --device auto
