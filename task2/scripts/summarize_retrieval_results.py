#!/usr/bin/env python3
"""
Summarize object-retrieval JSON outputs into compact metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize retrieval JSON metrics.")
    parser.add_argument(
        "--retrieval-jsons",
        nargs="+",
        type=Path,
        help="One or more retrieval JSON files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory to auto-discover object_retrieval_*.json files.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="K for mean@K statistics.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def score_at_k(rows: List[Dict[str, Any]], k: int) -> float:
    return mean([float(r["score"]) for r in rows[:k]])


def summarize_file(path: Path, top_k: int) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    query = data["query"]
    within = data.get("within_dataset_topk", [])
    cross = data.get("cross_dataset_topk", [])
    all_rows = data.get("all_candidates_sorted", [])

    summary: Dict[str, Any] = {
        "file": str(path),
        "query_dataset": query["dataset"],
        "query_slice": query["slice_id"],
        "query_rank": int(query["rank"]),
        "query_area": int(query["area"]),
        "num_objects_indexed": int(data.get("num_objects_indexed", 0)),
        "num_candidates": int(data.get("num_candidates", 0)),
        "within_top1": float(within[0]["score"]) if within else float("nan"),
        "cross_top1": float(cross[0]["score"]) if cross else float("nan"),
        "within_mean_at_k": score_at_k(within, top_k),
        "cross_mean_at_k": score_at_k(cross, top_k),
    }
    summary["mean_gap_within_minus_cross"] = summary["within_mean_at_k"] - summary["cross_mean_at_k"]

    per_cross_dataset: Dict[str, Dict[str, float]] = {}
    grouped: Dict[str, List[float]] = {}
    query_dataset = query["dataset"]
    for row in all_rows:
        ds = row["dataset"]
        if ds == query_dataset:
            continue
        grouped.setdefault(ds, []).append(float(row["score"]))

    for ds, vals in sorted(grouped.items()):
        vals_sorted = sorted(vals, reverse=True)
        per_cross_dataset[ds] = {
            "top1": vals_sorted[0],
            "mean_at_k": mean(vals_sorted[:top_k]),
            "mean_all": mean(vals_sorted),
            "count": len(vals_sorted),
        }
    summary["per_cross_dataset"] = per_cross_dataset
    return summary


def print_summary(s: Dict[str, Any], top_k: int) -> None:
    print("")
    print(f"File: {s['file']}")
    print(f"Query: {s['query_dataset']} | {s['query_slice']} | rank={s['query_rank']} | area={s['query_area']}")
    print(f"Within top1: {s['within_top1']:.3f}")
    print(f"Cross top1:  {s['cross_top1']:.3f}")
    print(f"Within mean@{top_k}: {s['within_mean_at_k']:.3f}")
    print(f"Cross mean@{top_k}:  {s['cross_mean_at_k']:.3f}")
    print(f"Gap (within-cross): {s['mean_gap_within_minus_cross']:.3f}")
    print("Cross by dataset:")
    per_ds = s["per_cross_dataset"]
    if not per_ds:
        print("  <none>")
        return
    for ds, m in per_ds.items():
        print(
            f"  {ds}: top1={m['top1']:.3f}, mean@{top_k}={m['mean_at_k']:.3f}, "
            f"mean_all={m['mean_all']:.3f}, n={m['count']}"
        )


def write_csv(summaries: List[Dict[str, Any]], top_k: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for s in summaries:
        base = {
            "file": s["file"],
            "query_dataset": s["query_dataset"],
            "query_slice": s["query_slice"],
            "query_rank": s["query_rank"],
            "query_area": s["query_area"],
            "num_objects_indexed": s["num_objects_indexed"],
            "num_candidates": s["num_candidates"],
            "within_top1": s["within_top1"],
            "cross_top1": s["cross_top1"],
            f"within_mean_at_{top_k}": s["within_mean_at_k"],
            f"cross_mean_at_{top_k}": s["cross_mean_at_k"],
            "mean_gap_within_minus_cross": s["mean_gap_within_minus_cross"],
            "cross_dataset": "",
            "cross_ds_top1": "",
            f"cross_ds_mean_at_{top_k}": "",
            "cross_ds_mean_all": "",
            "cross_ds_count": "",
        }
        rows.append(base)
        for ds, m in s["per_cross_dataset"].items():
            row = dict(base)
            row.update(
                {
                    "cross_dataset": ds,
                    "cross_ds_top1": m["top1"],
                    f"cross_ds_mean_at_{top_k}": m["mean_at_k"],
                    "cross_ds_mean_all": m["mean_all"],
                    "cross_ds_count": m["count"],
                }
            )
            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[ok] Wrote CSV summary to {out_path}")


def main() -> None:
    args = parse_args()
    paths: List[Path] = []
    if args.retrieval_jsons:
        paths.extend(args.retrieval_jsons)
    if args.results_dir:
        paths.extend(sorted(args.results_dir.glob("object_retrieval_*.json")))
    paths = sorted({p.resolve() for p in paths})
    if not paths:
        raise SystemExit("No retrieval JSONs found. Use --retrieval-jsons and/or --results-dir.")

    summaries = [summarize_file(p, args.top_k) for p in paths]
    for s in summaries:
        print_summary(s, args.top_k)

    if args.output_csv:
        write_csv(summaries, args.top_k, args.output_csv)


if __name__ == "__main__":
    main()

