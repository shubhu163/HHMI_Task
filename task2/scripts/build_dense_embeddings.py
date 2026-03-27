#!/usr/bin/env python3
"""
Build dense per-pixel embeddings from patch-token embeddings.

Method:
1. Reshape patch tokens to [C, grid_h, grid_w]
2. Bilinearly upsample to resized image resolution
3. Bilinearly upsample again to original image resolution
4. Optionally L2-normalize per pixel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DINO patch tokens into dense embeddings.")
    parser.add_argument("--emb-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dense_from_patch_tokens(
    patch_tokens: np.ndarray,
    grid_h: int,
    grid_w: int,
    resized_h: int,
    resized_w: int,
    orig_h: int,
    orig_w: int,
    l2_normalize: bool,
) -> np.ndarray:
    # patch_tokens: [N, C], where N = grid_h * grid_w
    c = patch_tokens.shape[1]
    feat = torch.from_numpy(patch_tokens.astype(np.float32)).reshape(grid_h, grid_w, c)
    feat = feat.permute(2, 0, 1).unsqueeze(0)  # [1, C, gh, gw]
    feat = F.interpolate(feat, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    feat = F.interpolate(feat, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    if l2_normalize:
        feat = F.normalize(feat, dim=1)
    return feat.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)  # [H, W, C]


def dense_from_tiled_patch_tokens(
    patch_tokens: np.ndarray,  # [T,N,C]
    tile_y: np.ndarray,
    tile_x: np.ndarray,
    tile_valid_h: np.ndarray,
    tile_valid_w: np.ndarray,
    grid_h: int,
    grid_w: int,
    tile_size: int,
    orig_h: int,
    orig_w: int,
    l2_normalize: bool,
) -> np.ndarray:
    c = int(patch_tokens.shape[-1])
    acc = torch.zeros((orig_h, orig_w, c), dtype=torch.float32)
    cnt = torch.zeros((orig_h, orig_w, 1), dtype=torch.float32)

    for i in range(patch_tokens.shape[0]):
        pt = patch_tokens[i]
        tile_dense = dense_from_patch_tokens(
            patch_tokens=pt,
            grid_h=grid_h,
            grid_w=grid_w,
            resized_h=tile_size,
            resized_w=tile_size,
            orig_h=tile_size,
            orig_w=tile_size,
            l2_normalize=False,
        )  # [tile,tile,C]

        y = int(tile_y[i])
        x = int(tile_x[i])
        vh = int(tile_valid_h[i])
        vw = int(tile_valid_w[i])
        tile_t = torch.from_numpy(tile_dense[:vh, :vw, :])
        acc[y : y + vh, x : x + vw, :] += tile_t
        cnt[y : y + vh, x : x + vw, :] += 1.0

    dense = acc / torch.clamp(cnt, min=1.0)
    if l2_normalize:
        dense = F.normalize(dense, dim=-1)
    return dense.numpy().astype(np.float32)


def process_dataset(dataset: str, args: argparse.Namespace) -> None:
    src_dir = args.emb_root / dataset
    out_dir = args.output_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_files = sorted(src_dir.glob("z*.npz"))
    if not emb_files:
        print(f"[skip] no embedding files in {src_dir}")
        return

    index_rows: List[Dict[str, object]] = []
    for emb_path in emb_files:
        out_path = out_dir / emb_path.name
        if out_path.exists() and not args.overwrite:
            continue

        data = np.load(emb_path, allow_pickle=True)
        patch_tokens = data["patch_tokens"]
        if patch_tokens.ndim == 2:
            dense = dense_from_patch_tokens(
                patch_tokens=patch_tokens,
                grid_h=int(data["grid_h"]),
                grid_w=int(data["grid_w"]),
                resized_h=int(data["resized_h"]),
                resized_w=int(data["resized_w"]),
                orig_h=int(data["orig_h"]),
                orig_w=int(data["orig_w"]),
                l2_normalize=args.l2_normalize,
            )
        elif patch_tokens.ndim == 3:
            dense = dense_from_tiled_patch_tokens(
                patch_tokens=patch_tokens,
                tile_y=data["tile_y"],
                tile_x=data["tile_x"],
                tile_valid_h=data["tile_valid_h"],
                tile_valid_w=data["tile_valid_w"],
                grid_h=int(data["grid_h"]),
                grid_w=int(data["grid_w"]),
                tile_size=int(data["tile_size"]),
                orig_h=int(data["orig_h"]),
                orig_w=int(data["orig_w"]),
                l2_normalize=args.l2_normalize,
            )
        else:
            raise RuntimeError(f"Unsupported patch_tokens shape: {patch_tokens.shape}")
        np.savez_compressed(
            out_path,
            dense=dense,
            source_embedding=str(emb_path),
            orig_h=int(data["orig_h"]),
            orig_w=int(data["orig_w"]),
        )
        index_rows.append(
            {
                "dataset": dataset,
                "slice_id": emb_path.stem,
                "dense_path": str(out_path),
                "source_embedding": str(emb_path),
            }
        )

    with (out_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index_rows, f, indent=2)
    print(f"[ok] {dataset}: dense embeddings in {out_dir}")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        process_dataset(dataset, args)


if __name__ == "__main__":
    main()
