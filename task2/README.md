# Microscopy Image Analysis 

This repository contains the implementation for the HHMI take-home challenge using OpenOrganelle EM data and DINOv3.

## Results Summary

We downloaded three datasets from OpenOrganelle: `jrc_hela-2`, `jrc_hela-3`, and `jrc_jurkat-1`. The design intentionally uses two same-lineage datasets (`jrc_hela-2`, `jrc_hela-3`) plus one distinct cell line (`jrc_jurkat-1`) to test whether embedding retrieval favors same cell type. To keep runtime practical while preserving organelle morphology, we used mito-containing 2D slices from scale `s2`, then ran tile-based DINOv3 embeddings, dense per-pixel feature reconstruction, and object-level retrieval.

Retrieval was analyzed in two rounds. Round 1 used query `jrc_hela-2:z00967:r0` against `jrc_hela-2` and `jrc_jurkat-1`; within-dataset retrieval was stronger than cross-dataset (`top1 0.946 vs 0.856`, `mean@10 0.932 vs 0.848`). Round 2 used query `jrc_hela-3:z00128:r0` against `jrc_hela-3`, `jrc_hela-2`, and `jrc_jurkat-1`; again within-dataset was strongest (`top1 0.946`, `mean@10 0.917`), and cross-dataset results showed clear same-lineage advantage: `jrc_hela-2` outperformed `jrc_jurkat-1` (`top1 0.880 vs 0.832`, `mean@10 0.865 vs 0.822`). Qualitatively, top-ranked panels preserve mitochondrial shape and internal texture better than bottom-ranked panels.

### Key Figures 

Round 1: Query `jrc_hela-2:z00967:r0`

![Round 1 Query](data/results/figures/object_retrieval_jrc_hela-2_z00967_r0/query.pdf)
![Round 1 Within Top-K](data/results/figures/object_retrieval_jrc_hela-2_z00967_r0/within_topk.pdf)
![Round 1 Cross Top-K (Jurkat)](data/results/figures/object_retrieval_jrc_hela-2_z00967_r0/cross_jrc_jurkat-1_topk.pdf)

Round 2: Query `jrc_hela-3:z00128:r0`

![Round 2 Query](data/results/figures/object_retrieval_jrc_hela-3_z00128_r0/query.pdf)
![Round 2 Cross Top-K (Hela-2)](data/results/figures/object_retrieval_jrc_hela-3_z00128_r0/cross_jrc_hela-2_topk.pdf)
![Round 2 Cross Top-K (Jurkat)](data/results/figures/object_retrieval_jrc_hela-3_z00128_r0/cross_jrc_jurkat-1_topk.pdf)

Metrics in this section can be reproduced with:

```bash
python ./scripts/summarize_retrieval_results.py \
  --results-dir ./data/results \
  --top-k 10 \
  --output-csv ./data/results/retrieval_summary.csv
```

## Setup

```
conda create -n task
conda activate task
pip install -r requirements.txt

```

Dependencies:
- numpy
- zarr<3
- fsspec[s3]
- torch
- timm
- torchmetrics
- scipy
- matplotlib

## Task 1 - Data Acquisition 

Implemented with:
- `scripts/download_zarr_subset.py`

Reference:
- OpenOrganelle FAQ data-access guidance (Python): use N5 data with `zarr` + `fsspec` style programmatic access from public S3. https://openorganelle.janelia.org/faq#data_access


Method:
- Open remote N5 datasets from OpenOrganelle public S3 with `zarr` (`N5FSStore`).
- Read raw EM and mitochondria segmentation arrays.
- Select mitochondria-containing 2D slices.
- Save paired `image/mask` slices plus metadata.
- Uses at least two datasets and is fully programmatic (no manual downloads).

Example:

```bash
python ./scripts/download_zarr_subset.py \
  --datasets jrc_hela-2 jrc_hela-3 jrc_jurkat-1 \
  --output-dir ./data/subset \
  --raw-key em/fibsem-uint16/s2 \
  --mito-key labels/mito_seg/s2 \
  --num-slices 24 \
  --min-mito-pixels 150 \
  --scan-step 4
```
Also written in the run_download_subset bash file so one can run that directly bash run_download_subset.sh


## Task 2 - DINO Feature Extraction

### 2.1 Embeddings

Implemented with:
- `scripts/extract_dino_embeddings.py`

Method:
- Pretrained DINOv3 model loaded via local `dinov3` repo and local `.pth` weights (`torch.hub`).
- Model weights are downloaded with `wget` from the Meta-provided URL after access approval/verification.
- We do tile-based inference (instead of resizing the full slice) to avoid aspect-ratio distortion and loss of local ultrastructure details.
- For each EM slice, we use tile size `384x384` and stride `192` (overlapping tiles).
- Each tile is converted to 3-channel input (grayscale repeated to RGB) and normalized with ImageNet mean/std before DINO inference.
- We use DINOv3 ViT-S+/16 (`dinov3_vits16plus`), where patch size is fixed at `16x16`.
- Therefore, each `384x384` tile becomes a `24x24` patch-token grid (`384/16 = 24`, total `576` tokens per tile).
- For every tile, we save patch tokens plus tile coordinates (`tile_y`, `tile_x`) and valid tile region size (for border tiles).
- Output is one `.npz` file per slice containing all tile tokens and metadata needed to reconstruct dense full-slice embeddings later.

Example:

```bash
python ./scripts/extract_dino_embeddings.py \
  --subset-root ./data/subset \
  --output-root ./data/embeddings \
  --datasets jrc_hela-2 jrc_hela-3 jrc_jurkat-1 \
  --hub-repo-dir ./dinov3 \
  --hub-entrypoint dinov3_vits16plus \
  --weights ./weights/dinov3_vits16_plus_pretrain_lvd1689m.pth \
  --batch-size 8 \
  --tile-size 384 \
  --tile-stride 192 \
  --device auto
```
Also written in the run_extract_embeddings.sh bash file so one can run that directly bash run_extract_embeddings.sh

### 2.2 Patch-size selection

DINOv3 ViT backbones used here are native `/16` variants.  
Therefore, patch-size discussion is handled as:
- native patch size = 16 (fixed for selected DINOv3 ViT backbones),
- effective granularity tuned via tile size, tile overlap, and source resolution level.

Selected practical setup:
- DINOv3 ViT-S+/16 (`dinov3_vits16plus`)
- tile size 384, stride 192, source scale `s2`.

- In ViT-S+/16, the model always reads the image as `16x16` pixel patches. We cannot change this to `8` or `32` without changing the backbone itself.
- So the practical “patch-size choice” for this task is about controlling the effective spatial detail around that fixed 16-pixel patch by:
  - choosing the source dataset scale (`s2` in our case),
  - choosing tile size (context per forward pass),
  - choosing tile stride (overlap between neighboring tiles).
- Why `s2`:
  - `s0` is full resolution but much heavier to process,
  - `s2` keeps useful mitochondria morphology while making extraction feasible on limited compute.
- Why tile size `384`:
  - gives enough local context for mitochondria shape/texture,
  - maps cleanly to `24x24` token grid (`384/16 = 24`).
- Why stride `192`:
  - 50% overlap between tiles improves spatial continuity,
  - reduces boundary artifacts when stitching dense embeddings.

### 2.3 Dense embeddings

Implemented with:
- `scripts/build_dense_embeddings.py`

Method:
- Input to this step is one embedding `.npz` per slice from Task 2.
- For each tile in that file:
  - patch tokens have shape `[N, C]`, where `N = grid_h * grid_w` (for our setup, `24*24 = 576`),
  - reshape tokens into a 2D token grid `[grid_h, grid_w, C]`,
  - upsample that grid to tile pixel size (`384x384`) to get a dense tile feature map `[384, 384, C]`.
- Each tile has stored coordinates (`tile_y`, `tile_x`) and valid area (`tile_valid_h`, `tile_valid_w`):
  - use these to place tile features at the correct location in the full slice,
  - crop padded border regions so only real image area contributes.
- Because tiles overlap (stride `192` on tile `384`), many pixels receive features from multiple tiles:
  - maintain an accumulator tensor (sum of features),
  - maintain a count tensor (number of contributing tiles per pixel),
  - final dense feature at each pixel = `sum / count`.
- Optional L2 normalization:
  - normalize each pixel embedding vector to unit length,
  - makes cosine similarity more stable for retrieval.
- Output:
  - one dense embedding file per slice with shape `[H, W, C]`,
  - these dense maps are then used for object-level pooling and retrieval in Task 3.

Example:

```bash
python ./scripts/build_dense_embeddings.py \
  --emb-root ./data/embeddings \
  --output-root ./data/dense_embeddings \
  --datasets jrc_hela-2 jrc_hela-3 jrc_jurkat-1 \
  --l2-normalize
```
Also written in the run_dense_embeddings.sh bash file so one can run that directly bash run_dense_embeddings.sh

## Task 3 - Embedding-Based Retrieval And Visualization

### 3.1 Single-query retrieval (implemented)

Implemented with:
- `scripts/retrieve_mito_objects.py`

Method:
- Extract connected mitochondria objects from each 2D mask.
- Pool dense features inside each object to get object embeddings.
- Use one object (`dataset + slice + rank`) as query.
- Compute cosine similarity against all other objects.

Output JSON includes:
- `within_dataset_topk`
- `cross_dataset_topk`
- `all_candidates_sorted` (all scored candidates)

Example:

```bash
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
```
Also written in the run_retrieval_objects.sh bash file so one can run that directly bash run_retrieval_objects.sh
This run script executes two retrieval rounds: `jrc_hela-2:z00967:r0` and `jrc_hela-3:z00128:r0`.


### 3.2 Visualizations (implemented)

Implemented with:
- `scripts/visualize_retrieval.py`

Current visualization outputs:
- query object panel
- within-dataset top-k
- within-dataset bottom-k
- cross-dataset top-k (all non-query datasets combined)
- cross-dataset bottom-k (all non-query datasets combined)
- per-dataset cross comparison panels (e.g., query `jrc_hela-3` vs `jrc_hela-2`, and vs `jrc_jurkat-1`)

Each panel shows a cropped EM region, and the queried/retrieved mitochondrion is highlighted in red; the score corresponds to that highlighted object.

Example:

```bash
python ./scripts/visualize_retrieval.py \
  --subset-root ./data/subset \
  --retrieval-json ./data/results/object_retrieval_jrc_hela-3_z00128_r0.json \
  --output-dir ./data/results/figures/object_retrieval_jrc_hela-3_z00128_r0 \
  --top-k 10 \
  --bottom-k 10 \
  --margin 24 \
  --font-size 16 \
  --max-cols 2
```
Also written in the run_visualize_retrieval.sh bash file so one can run that directly bash run_visualize_retrieval.sh

### 3.3 Multiple Queries (Proposal-Only)

For multiple query mitochondria, we would pick a few query objects instead of only one. Then we would get one embedding from each query object and combine them into one average query embedding (mean prototype).

How retrieval would work:
- get one embedding per query mitochondrion,
- average them into one final query vector,
- compare all candidate mitochondria to this final query using cosine similarity,
- rank candidates by score and compare with single-query results.

How visualization would change:
- keep current within/cross top-k and bottom-k plots,
- add one plot per query object (to see individual behavior),
- add one combined plot from the averaged query.

What we expect:
- results become more stable,
- less sensitivity to one bad/noisy query,
- broader matches across mitochondria shapes,
- but sometimes slightly less sharp matching to one exact morphology.

## Task 4 - Proposal: Minimal Fine-Tuning

Goal:
- improve mitochondria detection/segmentation over the current zero-shot setup,
- while training as few parameters as possible.

Model plan:
- keep the DINOv3 backbone frozen,
- add a small trainable module on top:
  - Option A: LoRA/adapters in a few late blocks,
  - Option B: a lightweight segmentation head that takes dense features.

Training data plan:
- train on multiple datasets together (same-lineage and different-lineage cell types),
- split by slice into train/val/test (for example 70/15/15),
- apply simple augmentations (flip, small rotation, intensity jitter).

Training objective:
- optimize Dice + BCE loss for mitochondria mask prediction,
- use early stopping on validation Dice,
- keep learning rate small because backbone stays mostly frozen.

Evaluation plan:
- segmentation quality: Dice, IoU, precision, recall on test slices,
- retrieval quality: compare top-1 and mean@10 before vs after fine-tuning,
- cross-dataset check: verify if same-lineage (`hela-2`/`hela-3`) and different-lineage (`jurkat-1`) behavior improves consistently.

Why this is a good fit:
- much lower compute than full fine-tuning,
- fewer trainable parameters and lower overfitting risk,
- keeps strong pretrained visual features,
- still adapts features to EM mitochondria structure,
- and improves robustness across different cell lines by training on mixed datasets.

## Reproduction Order

Use run scripts in this order:

1. `bash run_download_subset.sh`
2. `bash run_extract_embeddings.sh`
3. `bash run_dense_embeddings.sh`
4. `bash run_retrieval_objects.sh`
5. `bash run_visualize_retrieval.sh`

## Notes

- If a retrieval query object is missing, reduce `--min-object-pixels` or pick a different query rank/slice.

## Repo Structure

Expected layout after running the bash pipeline:

```text
task2/
  README.md
  requirements.txt
  task.md
  run_download_subset.sh
  run_extract_embeddings.sh
  run_dense_embeddings.sh
  run_retrieval_objects.sh
  run_visualize_retrieval.sh
  scripts/
    download_zarr_subset.py
    extract_dino_embeddings.py
    build_dense_embeddings.py
    retrieve_mito_objects.py
    visualize_retrieval.py
    summarize_retrieval_results.py
  data/
    subset/
      jrc_hela-2/
      jrc_hela-3/
      jrc_jurkat-1/
    embeddings/
      jrc_hela-2/
      jrc_hela-3/
      jrc_jurkat-1/
    dense_embeddings/
      jrc_hela-2/
      jrc_hela-3/
      jrc_jurkat-1/
    results/
      object_retrieval_*.json
      figures/
```

## Large Data Note

Due to repository size limits, large generated artifacts are not pushed:
- `data/subset/`
- `data/embeddings/`
- `data/dense_embeddings/`

These can be regenerated using the run scripts in the Reproduction Order section.

In our current run configuration, we download `24` slices per dataset (`--num-slices 24`) by scanning the mitochondria segmentation mask (`labels/mito_seg/s2`) and selecting slices that pass the minimum mitochondria-pixel threshold.


## AI Assistance Disclosure
Codex was used to accelerate implementation, including drafting portions of the code, script editing, debugging support, and README organization. The overall approach, dataset selection, experimental decisions, and final validation of the submission were completed by me.

