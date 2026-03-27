#!/usr/bin/env python3
"""
Extract DINOv3 patch embeddings from EM slices using tile-based inference.

This avoids full-slice resizing distortion by:
1) splitting each slice into overlapping tiles
2) running the DINOv3 backbone per tile
3) storing tile patch tokens + tile coordinates per slice
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINOv3 tile embeddings for EM slices.")
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--hub-repo-dir", type=Path, required=True, help="Path to local cloned dinov3 repo.")
    parser.add_argument("--hub-entrypoint", default="dinov3_vits16")
    parser.add_argument("--weights", required=True, help="Path or URL to DINOv3 backbone weights.")
    parser.add_argument("--batch-size", type=int, default=8, help="Tile batch size.")
    parser.add_argument("--tile-size", type=int, default=384, help="Tile size in pixels.")
    parser.add_argument("--tile-stride", type=int, default=192, help="Tile stride in pixels.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_em(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lo = np.percentile(image, 1.0)
    hi = np.percentile(image, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def to_rgb_float(image: np.ndarray) -> np.ndarray:
    gray = normalize_em(image)
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)  # [H,W,3] in [0,1]


def infer_patch_size(model: torch.nn.Module) -> int:
    patch = model.patch_embed.patch_size
    if isinstance(patch, tuple):
        return int(patch[0])
    return int(patch)


def compute_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_image(image_rgb: np.ndarray, tile_size: int, stride: int) -> List[Dict[str, object]]:
    h, w, _ = image_rgb.shape
    ys = compute_starts(h, tile_size, stride)
    xs = compute_starts(w, tile_size, stride)
    tiles: List[Dict[str, object]] = []
    for y in ys:
        for x in xs:
            crop = image_rgb[y : y + tile_size, x : x + tile_size, :]
            valid_h, valid_w = crop.shape[0], crop.shape[1]
            if valid_h < tile_size or valid_w < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
                padded[:valid_h, :valid_w, :] = crop
                crop = padded
            tiles.append(
                {
                    "tile": crop,
                    "y": y,
                    "x": x,
                    "valid_h": valid_h,
                    "valid_w": valid_w,
                }
            )
    return tiles


def build_inputs(rgb_batch: List[np.ndarray], device: torch.device) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    for img in rgb_batch:
        t = torch.from_numpy(img).permute(2, 0, 1).float()  # [3,H,W], already [0,1]
        t = (t - mean) / std
        tensors.append(t)
    return torch.stack(tensors, dim=0).to(device)


def extract_tokens(model: torch.nn.Module, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        feat = model.forward_features(pixel_values)
    if isinstance(feat, dict):
        patch = feat.get("x_norm_patchtokens")
        cls = feat.get("x_norm_clstoken")
        if patch is None:
            raise RuntimeError("TorchHub model did not return x_norm_patchtokens.")
        if cls is None:
            cls = patch.mean(dim=1)
        return patch, cls
    if torch.is_tensor(feat):
        return feat[:, 1:, :], feat[:, 0, :]
    raise RuntimeError("Unsupported forward_features output format.")


def save_slice_embedding(
    out_path: Path,
    patch_tokens: np.ndarray,  # [T, N, C]
    cls_tokens: np.ndarray,  # [T, C]
    tile_y: np.ndarray,
    tile_x: np.ndarray,
    tile_valid_h: np.ndarray,
    tile_valid_w: np.ndarray,
    grid_h: int,
    grid_w: int,
    tile_size: int,
    patch_size: int,
    source_path: Path,
    orig_h: int,
    orig_w: int,
) -> None:
    np.savez_compressed(
        out_path,
        patch_tokens=patch_tokens.astype(np.float32),
        cls_token=cls_tokens.astype(np.float32),
        tile_y=tile_y.astype(np.int32),
        tile_x=tile_x.astype(np.int32),
        tile_valid_h=tile_valid_h.astype(np.int32),
        tile_valid_w=tile_valid_w.astype(np.int32),
        grid_h=np.int32(grid_h),
        grid_w=np.int32(grid_w),
        tile_size=np.int32(tile_size),
        patch_h=np.int32(patch_size),
        patch_w=np.int32(patch_size),
        orig_h=np.int32(orig_h),
        orig_w=np.int32(orig_w),
        resized_h=np.int32(tile_size),
        resized_w=np.int32(tile_size),
        source_image=str(source_path),
    )


def run_dataset(
    dataset: str,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int,
) -> None:
    image_dir = args.subset_root / dataset / "images"
    if not image_dir.exists():
        print(f"[skip] Missing image directory: {image_dir}")
        return

    out_dir = args.output_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_dir.glob("*.npy"))
    if not image_paths:
        print(f"[skip] No image files in {image_dir}")
        return

    if args.tile_size % patch_size != 0:
        raise ValueError(f"--tile-size ({args.tile_size}) must be divisible by patch size ({patch_size}).")

    grid_h = args.tile_size // patch_size
    grid_w = args.tile_size // patch_size
    expected_tokens = grid_h * grid_w

    index_rows: List[Dict[str, object]] = []

    for src_path in image_paths:
        stem = src_path.stem
        out_path = out_dir / f"{stem}.npz"
        if out_path.exists() and not args.overwrite:
            continue

        image = np.load(src_path)
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
        rgb = to_rgb_float(image)
        tiles = tile_image(rgb, tile_size=args.tile_size, stride=args.tile_stride)

        tile_patch_list: List[np.ndarray] = []
        tile_cls_list: List[np.ndarray] = []
        tile_y: List[int] = []
        tile_x: List[int] = []
        tile_valid_h: List[int] = []
        tile_valid_w: List[int] = []

        for i in range(0, len(tiles), args.batch_size):
            batch_tiles = tiles[i : i + args.batch_size]
            rgb_batch = [t["tile"] for t in batch_tiles]
            pixel_values = build_inputs(rgb_batch, device)
            patch_tokens, cls_tokens = extract_tokens(model, pixel_values)
            if patch_tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    f"Token/grid mismatch: got {patch_tokens.shape[1]} tokens but expected {expected_tokens} "
                    f"for tile_size={args.tile_size}, patch_size={patch_size}."
                )

            pt = patch_tokens.cpu().numpy()
            ct = cls_tokens.cpu().numpy()
            for j, meta in enumerate(batch_tiles):
                tile_patch_list.append(pt[j])
                tile_cls_list.append(ct[j])
                tile_y.append(int(meta["y"]))
                tile_x.append(int(meta["x"]))
                tile_valid_h.append(int(meta["valid_h"]))
                tile_valid_w.append(int(meta["valid_w"]))

        save_slice_embedding(
            out_path=out_path,
            patch_tokens=np.stack(tile_patch_list, axis=0),
            cls_tokens=np.stack(tile_cls_list, axis=0),
            tile_y=np.asarray(tile_y),
            tile_x=np.asarray(tile_x),
            tile_valid_h=np.asarray(tile_valid_h),
            tile_valid_w=np.asarray(tile_valid_w),
            grid_h=grid_h,
            grid_w=grid_w,
            tile_size=args.tile_size,
            patch_size=patch_size,
            source_path=src_path,
            orig_h=orig_h,
            orig_w=orig_w,
        )

        index_rows.append(
            {
                "dataset": dataset,
                "slice_id": stem,
                "embedding_path": str(out_path),
                "source_image": str(src_path),
                "num_tiles": len(tiles),
                "tile_size": args.tile_size,
                "tile_stride": args.tile_stride,
                "grid_h": int(grid_h),
                "grid_w": int(grid_w),
                "hub_entrypoint": args.hub_entrypoint,
                "weights": args.weights,
            }
        )

    with (out_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index_rows, f, indent=2)
    print(f"[ok] {dataset}: wrote tiled embeddings to {out_dir}")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    repo_dir = str(args.hub_repo_dir.resolve())
    model = torch.hub.load(
        repo_dir,
        args.hub_entrypoint,
        source="local",
        weights=args.weights,
    ).eval().to(device)
    patch_size = infer_patch_size(model)

    for dataset in args.datasets:
        run_dataset(dataset, args, model, device, patch_size)


if __name__ == "__main__":
    main()
