#!/usr/bin/env python3
"""
Object-level mitochondria retrieval from dense embeddings.

This script:
1) Extracts connected mitochondria objects from 2D masks
2) Pools dense features per object
3) Runs cosine retrieval from a single query mitochondrion
4) Saves a JSON report for visualization and analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object-level mito retrieval from dense embeddings.")
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--query-dataset", required=True)
    parser.add_argument("--query-slice", required=True, help="e.g. z00198")
    parser.add_argument(
        "--query-rank",
        type=int,
        default=0,
        help="Object rank by area within query slice (0 = largest mito object).",
    )
    parser.add_argument("--min-object-pixels", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def extract_objects(mask: np.ndarray, min_pixels: int) -> List[Dict[str, object]]:
    binary = mask > 0
    labels, n = ndi.label(binary)
    objects: List[Dict[str, object]] = []
    for label_id in range(1, n + 1):
        ys, xs = np.where(labels == label_id)
        if ys.size == 0:
            continue
        area = int(ys.size)
        if area < min_pixels:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        objects.append(
            {
                "label_id": label_id,
                "area": area,
                "bbox": [y0, y1, x0, x1],
            }
        )
    objects.sort(key=lambda x: x["area"], reverse=True)
    return objects


def pooled_object_embedding(dense: np.ndarray, labels: np.ndarray, label_id: int) -> np.ndarray:
    vecs = dense[labels == label_id]
    if vecs.size == 0:
        return dense.reshape(-1, dense.shape[-1]).mean(axis=0).astype(np.float32)
    return vecs.mean(axis=0).astype(np.float32)


def build_catalog(
    subset_root: Path,
    dense_root: Path,
    datasets: List[str],
    min_object_pixels: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        dense_dir = dense_root / dataset
        mask_dir = subset_root / dataset / "masks"
        if not dense_dir.exists() or not mask_dir.exists():
            continue
        for dense_path in sorted(dense_dir.glob("z*.npz")):
            slice_id = dense_path.stem
            mask_path = mask_dir / f"{slice_id}.npy"
            if not mask_path.exists():
                continue

            data = np.load(dense_path, allow_pickle=True)
            dense = data["dense"]  # [H,W,C]
            mask = np.load(mask_path)
            labels, _ = ndi.label(mask > 0)
            objects = extract_objects(mask, min_pixels=min_object_pixels)

            for rank, obj in enumerate(objects):
                label_id = int(obj["label_id"])
                emb = pooled_object_embedding(dense, labels, label_id)
                rows.append(
                    {
                        "dataset": dataset,
                        "slice_id": slice_id,
                        "rank": rank,
                        "label_id": label_id,
                        "area": int(obj["area"]),
                        "bbox": [int(v) for v in obj["bbox"]],
                        "embedding": emb,
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    rows = build_catalog(
        subset_root=args.subset_root,
        dense_root=args.dense_root,
        datasets=args.datasets,
        min_object_pixels=args.min_object_pixels,
    )
    if not rows:
        raise RuntimeError("No mitochondria objects found.")

    query = None
    for row in rows:
        if (
            row["dataset"] == args.query_dataset
            and row["slice_id"] == args.query_slice
            and row["rank"] == args.query_rank
        ):
            query = row
            break
    if query is None:
        raise RuntimeError(
            f"Query object not found for dataset={args.query_dataset}, "
            f"slice={args.query_slice}, rank={args.query_rank}."
        )

    qvec = query["embedding"]
    scored: List[Dict[str, object]] = []
    for row in rows:
        if (
            row["dataset"] == query["dataset"]
            and row["slice_id"] == query["slice_id"]
            and row["rank"] == query["rank"]
        ):
            continue
        score = cosine_similarity(qvec, row["embedding"])
        scored.append(
            {
                "dataset": row["dataset"],
                "slice_id": row["slice_id"],
                "rank": int(row["rank"]),
                "label_id": int(row["label_id"]),
                "area": int(row["area"]),
                "bbox": [int(v) for v in row["bbox"]],
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    within = [r for r in scored if r["dataset"] == args.query_dataset][: args.top_k]
    cross = [r for r in scored if r["dataset"] != args.query_dataset][: args.top_k]

    result = {
        "query": {
            "dataset": query["dataset"],
            "slice_id": query["slice_id"],
            "rank": int(query["rank"]),
            "label_id": int(query["label_id"]),
            "area": int(query["area"]),
            "bbox": [int(v) for v in query["bbox"]],
        },
        "within_dataset_topk": within,
        "cross_dataset_topk": cross,
        "all_candidates_sorted": scored,
        "num_objects_indexed": len(rows),
        "num_candidates": len(scored),
        "min_object_pixels": args.min_object_pixels,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[ok] Wrote object retrieval results to {args.output_json}")


if __name__ == "__main__":
    main()
