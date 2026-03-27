#!/usr/bin/env bash
set -euo pipefail

python ./scripts/download_zarr_subset.py \
  --datasets \
    jrc_jurkat-1 jrc_hela-3 jrc_hela-2 \
  --output-dir ./data/subset \
  --raw-key em/fibsem-uint16/s2 \
  --mito-key labels/mito_seg/s2 \
  --num-slices 24 \
  --min-mito-pixels 150 \
  --scan-step 4
