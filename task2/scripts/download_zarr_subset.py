#!/usr/bin/env python3
r"""
Download mitochondria-containing 2D slices from OpenOrganelle N5/Zarr data
using the FAQ-recommended Python stack: fsspec + zarr.

Example:
    python scripts/download_zarr_subset.py \
        --datasets jrc_hela-2 jrc_jurkat-1 \
        --output-dir data/subset \
        --mito-key labels/mito_seg/s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import zarr


DEFAULT_BUCKET = "s3://janelia-cosem-datasets"
DEFAULT_RAW_KEY = "em/fibsem-uint16/s0"
DEFAULT_MITO_KEY = "labels/mito_seg/s0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download mito-containing slices from OpenOrganelle N5/Zarr datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names, e.g. jrc_hela-2 jrc_jurkat-1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where extracted slices and metadata will be saved.",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket prefix. Defaults to {DEFAULT_BUCKET}.",
    )
    parser.add_argument(
        "--container-template",
        default="{bucket}/{dataset}/{dataset}.n5",
        help="Template used to build the remote N5 path.",
    )
    parser.add_argument(
        "--raw-key",
        default=DEFAULT_RAW_KEY,
        help=f"Array key for the raw EM volume. Defaults to {DEFAULT_RAW_KEY}.",
    )
    parser.add_argument(
        "--mito-key",
        default=DEFAULT_MITO_KEY,
        help=f"Array key for the mitochondria labels. Defaults to {DEFAULT_MITO_KEY}.",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=20,
        help="Maximum number of qualifying z-slices to save per dataset.",
    )
    parser.add_argument(
        "--scan-step",
        type=int,
        default=1,
        help="Evaluate every Nth z-slice when searching candidates (higher is faster).",
    )
    parser.add_argument(
        "--min-mito-pixels",
        type=int,
        default=500,
        help="Minimum nonzero mito pixels required for a slice to qualify.",
    )
    parser.add_argument(
        "--save-format",
        choices=["npy", "npz"],
        default="npy",
        help="Output format for saved arrays.",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="Print available group and array paths for each dataset and exit without saving slices.",
    )
    return parser.parse_args()


def build_container_url(bucket: str, dataset: str, template: str) -> str:
    return template.format(bucket=bucket.rstrip("/"), dataset=dataset)


def open_n5_group(url: str) -> zarr.Group:
    n5_store_cls = getattr(zarr, "N5FSStore", None)
    if n5_store_cls is None:
        raise RuntimeError(
            "This script requires zarr 2.x with N5FSStore support. "
            "Please install 'zarr<3'."
        )
    store = n5_store_cls(url, anon=True)
    return zarr.open(store, mode="r")


def open_array(group: zarr.Group, key: str) -> zarr.Array:
    if key not in group:
        available_paths = collect_group_paths(group)
        key_tail = key.split("/")[-1]
        likely_matches = [
            path
            for path in available_paths
            if key_tail in path or "fibsem" in path or "em" in path or "mito" in path
        ]
        match_preview = "\n".join(f"  {path}" for path in likely_matches[:50])
        raise KeyError(
            f"Array key '{key}' not found.\n"
            f"Likely matches:\n{match_preview if match_preview else '  <none found>'}"
        )
    array = group[key]
    if not isinstance(array, zarr.Array):
        raise TypeError(f"Key '{key}' is not a zarr array.")
    return array


def validate_pair(dataset: str, raw_array: zarr.Array, mito_array: zarr.Array) -> int:
    if raw_array.ndim != 3 or mito_array.ndim != 3:
        raise ValueError(
            f"{dataset}: expected 3D arrays, got raw ndim={raw_array.ndim}, mito ndim={mito_array.ndim}."
        )
    if raw_array.shape[1:] != mito_array.shape[1:]:
        raise ValueError(
            f"{dataset}: shape mismatch raw={raw_array.shape} mito={mito_array.shape}."
        )
    usable_z = min(raw_array.shape[0], mito_array.shape[0])
    if raw_array.shape[0] != mito_array.shape[0]:
        print(
            f"[warn] {dataset}: z-depth mismatch raw={raw_array.shape[0]} mito={mito_array.shape[0]}; "
            f"using first {usable_z} slices where both are aligned."
        )
    return usable_z


def count_positive_pixels_per_slice(
    mito_array: zarr.Array, scan_step: int, max_z: int
) -> Dict[int, int]:
    step = max(1, int(scan_step))
    counts: Dict[int, int] = {}
    for z_idx in range(0, max_z, step):
        counts[z_idx] = int(np.count_nonzero(np.asarray(mito_array[z_idx])))
    return counts


def choose_evenly_spread_slices(
    counts_by_z: Dict[int, int], min_mito_pixels: int, num_slices: int
) -> List[int]:
    valid = [idx for idx, count in counts_by_z.items() if count >= min_mito_pixels]
    valid.sort()
    if len(valid) <= num_slices:
        return valid
    positions = np.linspace(0, len(valid) - 1, num=num_slices, dtype=int)
    return [valid[pos] for pos in positions]


def save_array(path: Path, array: np.ndarray, save_format: str) -> None:
    if save_format == "npy":
        np.save(path, array)
    else:
        np.savez_compressed(path, array=array)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def collect_group_paths(group: zarr.Group, prefix: str = "") -> List[str]:
    paths: List[str] = []

    for group_name in sorted(group.group_keys()):
        child_path = f"{prefix}/{group_name}" if prefix else group_name
        paths.append(f"{child_path}/")
        child = group[group_name]
        if isinstance(child, zarr.Group):
            paths.extend(collect_group_paths(child, child_path))

    for array_name in sorted(group.array_keys()):
        array_path = f"{prefix}/{array_name}" if prefix else array_name
        paths.append(array_path)

    return paths


def process_dataset(args: argparse.Namespace, dataset: str) -> None:
    container_url = build_container_url(args.bucket, dataset, args.container_template)
    print(f"\nDataset: {dataset}")
    print(f"Container: {container_url}")

    group = open_n5_group(container_url)

    if args.list_keys:
        print("Available group/array paths:")
        for key in collect_group_paths(group):
            print(f"  {key}")
        return

    raw_array = open_array(group, args.raw_key)
    mito_array = open_array(group, args.mito_key)
    usable_z = validate_pair(dataset, raw_array, mito_array)

    counts = count_positive_pixels_per_slice(
        mito_array, scan_step=args.scan_step, max_z=usable_z
    )
    selected_slices = choose_evenly_spread_slices(
        counts, min_mito_pixels=args.min_mito_pixels, num_slices=args.num_slices
    )

    dataset_dir = args.output_dir / dataset
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".npy" if args.save_format == "npy" else ".npz"
    manifest: List[Dict[str, Any]] = []

    for z_idx in selected_slices:
        raw_slice = np.asarray(raw_array[z_idx])
        mito_slice = (np.asarray(mito_array[z_idx]) > 0).astype(np.uint8)

        image_path = images_dir / f"z{z_idx:05d}{suffix}"
        mask_path = masks_dir / f"z{z_idx:05d}{suffix}"

        save_array(image_path, raw_slice, args.save_format)
        save_array(mask_path, mito_slice, args.save_format)

        manifest.append(
            {
                "dataset": dataset,
                "z_index": z_idx,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "mito_pixels": counts[z_idx],
                "image_shape": list(raw_slice.shape),
            }
        )

    write_json(
        dataset_dir / "metadata.json",
        {
            "dataset": dataset,
            "container_url": container_url,
            "raw_key": args.raw_key,
            "mito_key": args.mito_key,
            "raw_shape": list(raw_array.shape),
            "mito_shape": list(mito_array.shape),
            "usable_z": usable_z,
            "selected_slices": selected_slices,
            "num_selected_slices": len(selected_slices),
            "min_mito_pixels": args.min_mito_pixels,
            "scan_step": args.scan_step,
        },
    )
    write_json(dataset_dir / "manifest.json", {"slices": manifest})
    print(f"Saved {len(selected_slices)} slices to {dataset_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        process_dataset(args, dataset)


if __name__ == "__main__":
    main()
