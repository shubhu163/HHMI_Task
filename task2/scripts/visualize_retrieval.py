#!/usr/bin/env python3
"""
Visualize object-level retrieval results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

FONT_SIZE = 16

# Enforce consistent readable typography across all saved figures.
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize mitochondria retrieval panels.")
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--retrieval-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=10)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--max-cols", type=int, default=3, help="Maximum number of panels per row.")
    parser.add_argument("--font-size", type=int, default=16, help="Font size for subplot and figure titles.")
    return parser.parse_args()


def load_slice(subset_root: Path, dataset: str, slice_id: str) -> Tuple[np.ndarray, np.ndarray]:
    img = np.load(subset_root / dataset / "images" / f"{slice_id}.npy")
    msk = np.load(subset_root / dataset / "masks" / f"{slice_id}.npy")
    return img, msk


def make_object_mask(mask: np.ndarray, label_id: int) -> np.ndarray:
    labels, _ = ndi.label(mask > 0)
    return labels == int(label_id)


def crop_with_margin(image: np.ndarray, obj_mask: np.ndarray, bbox: List[int], margin: int) -> Tuple[np.ndarray, np.ndarray]:
    y0, y1, x0, x1 = bbox
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(image.shape[0], y1 + margin)
    x1 = min(image.shape[1], x1 + margin)
    return image[y0:y1, x0:x1], obj_mask[y0:y1, x0:x1]


def draw_overlay(ax, image: np.ndarray, obj_mask: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray")
    overlay = np.zeros((*obj_mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = obj_mask.astype(np.float32) * 0.35
    ax.imshow(overlay)
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.axis("off")


def panel(
    items: List[Dict[str, object]],
    subset_root: Path,
    title: str,
    out_path: Path,
    top_k: int,
    margin: int,
    max_cols: int,
) -> None:
    items = items[:top_k]
    n = len(items)
    cols = min(max(1, int(max_cols)), max(1, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        item = items[i]
        img, msk = load_slice(subset_root, item["dataset"], item["slice_id"])
        obj = make_object_mask(msk, int(item["label_id"]))
        crop_img, crop_obj = crop_with_margin(img, obj, item["bbox"], margin)
        score = item.get("score", None)
        if score is None:
            t = f"{item['dataset']}:{item['slice_id']} r{item['rank']}"
        else:
            t = f"{item['dataset']}:{item['slice_id']} r{item['rank']} s={score:.3f}"
        draw_overlay(ax, crop_img, crop_obj, t)

    fig.suptitle(title, fontsize=FONT_SIZE + 2, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    global FONT_SIZE
    FONT_SIZE = int(args.font_size)
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
        }
    )
    with args.retrieval_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    q = data["query"]
    query_item = [q]
    within = data["within_dataset_topk"]
    cross = data["cross_dataset_topk"]
    all_candidates = data.get("all_candidates_sorted", [])
    all_within = [r for r in all_candidates if r["dataset"] == q["dataset"]]
    all_cross = [r for r in all_candidates if r["dataset"] != q["dataset"]]
    bottom_within = list(reversed(all_within[-args.bottom_k:])) if all_within else []
    bottom_cross = list(reversed(all_cross[-args.bottom_k:])) if all_cross else []
    cross_datasets = sorted({r["dataset"] for r in all_cross})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_path = args.output_dir / "query.pdf"
    within_top_path = args.output_dir / "within_topk.pdf"
    within_bottom_path = args.output_dir / "within_bottomk.pdf"
    cross_top_path = args.output_dir / "cross_all_topk.pdf"
    cross_bottom_path = args.output_dir / "cross_all_bottomk.pdf"

    panel(query_item, args.subset_root, "Query Mitochondrion", query_path, top_k=1, margin=args.margin, max_cols=args.max_cols)
    panel(within, args.subset_root, "Within-Dataset Top-K", within_top_path, top_k=args.top_k, margin=args.margin, max_cols=args.max_cols)
    panel(
        bottom_within,
        args.subset_root,
        "Within-Dataset Bottom-K",
        within_bottom_path,
        top_k=args.bottom_k,
        margin=args.margin,
        max_cols=args.max_cols,
    )
    panel(cross, args.subset_root, "Cross-Dataset Top-K (All Other Datasets)", cross_top_path, top_k=args.top_k, margin=args.margin, max_cols=args.max_cols)
    panel(
        bottom_cross,
        args.subset_root,
        "Cross-Dataset Bottom-K (All Other Datasets)",
        cross_bottom_path,
        top_k=args.bottom_k,
        margin=args.margin,
        max_cols=args.max_cols,
    )

    for dataset in cross_datasets:
        ds_items = [r for r in all_cross if r["dataset"] == dataset]
        ds_bottom = list(reversed(ds_items[-args.bottom_k:])) if ds_items else []
        safe_name = dataset.replace("/", "_")
        panel(
            ds_items,
            args.subset_root,
            f"Cross-Dataset Top-K: query {q['dataset']} vs {dataset}",
            args.output_dir / f"cross_{safe_name}_topk.pdf",
            top_k=args.top_k,
            margin=args.margin,
            max_cols=args.max_cols,
        )
        panel(
            ds_bottom,
            args.subset_root,
            f"Cross-Dataset Bottom-K: query {q['dataset']} vs {dataset}",
            args.output_dir / f"cross_{safe_name}_bottomk.pdf",
            top_k=args.bottom_k,
            margin=args.margin,
            max_cols=args.max_cols,
        )
    print(f"[ok] Wrote panels to {args.output_dir}")


if __name__ == "__main__":
    main()
