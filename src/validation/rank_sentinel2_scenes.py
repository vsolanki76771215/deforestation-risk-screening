#!/usr/bin/env python3
"""Rank Sentinel-2 scenes using SCL-valid AOI coverage and select composites."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds, transform_geom


RANKED_FIELDS = [
    "aoi_id", "aoi_role", "target_year", "rank", "item_id",
    "acquisition_datetime", "cloud_cover_pct", "aoi_pixel_count",
    "scene_footprint_pixel_count", "valid_aoi_pixel_count",
    "footprint_coverage_pct", "valid_coverage_pct", "invalid_coverage_pct",
    "ranking_score", "meets_minimum_valid_coverage", "selected_for_composite",
    "selection_order", "incremental_valid_pixels", "cumulative_valid_coverage_pct",
    "SCL_href", "status", "reason",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sas_token(catalog_url: str, collection: str) -> str:
    parsed = urllib.parse.urlsplit(catalog_url)
    url = f"{parsed.scheme}://{parsed.netloc}/api/sas/v1/token/{collection}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["token"]


def signed_href(href: str, token: str) -> str:
    separator = "&" if "?" in href else "?"
    return f"{href}{separator}{token}"


def target_grid(bounds: list[float], dst_crs: str, resolution: float):
    left, bottom, right, top = transform_bounds("EPSG:4326", dst_crs, *bounds)
    left = math.floor(left / resolution) * resolution
    bottom = math.floor(bottom / resolution) * resolution
    right = math.ceil(right / resolution) * resolution
    top = math.ceil(top / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    transform = from_origin(left, top, resolution, resolution)
    polygon = {"type": "Polygon", "coordinates": [[
        [bounds[0], bounds[1]], [bounds[2], bounds[1]],
        [bounds[2], bounds[3]], [bounds[0], bounds[3]],
        [bounds[0], bounds[1]],
    ]]}
    dst_polygon = transform_geom("EPSG:4326", dst_crs, polygon)
    aoi_mask = geometry_mask([dst_polygon], out_shape=(height, width),
                             transform=transform, invert=True)
    return transform, width, height, aoi_mask


def scene_masks(href: str, token: str, dst_crs: str, dst_transform,
                width: int, height: int, aoi_mask: np.ndarray,
                excluded: set[int]) -> tuple[np.ndarray, np.ndarray]:
    for attempt in range(3):
        scl = np.full((height, width), 255, dtype=np.uint8)
        try:
            with rasterio.Env(GDAL_HTTP_UNSAFESSL="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open(signed_href(href, token)) as src:
                    reproject(
                        source=rasterio.band(src, 1), destination=scl,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=dst_transform, dst_crs=dst_crs,
                        src_nodata=0, dst_nodata=255, resampling=Resampling.nearest,
                    )
            break
        except rasterio.errors.RasterioError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    footprint = aoi_mask & (scl != 255)
    valid = footprint & ~np.isin(scl, list(excluded))
    return footprint, valid


def rank_and_select(rows: list[dict[str, str]], config: dict[str, Any], token: str):
    s2 = config["sentinel2"]
    policy = s2["ranking"]
    dst_crs = config["specification"]["analysis_crs"]
    minimum = float(s2["minimum_valid_coverage_pct"])
    excluded = set(map(int, s2["excluded_scl_classes"]))
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["aoi_id"], int(row["target_year"]))].append(row)

    ranked_output: list[dict[str, Any]] = []
    selected_output: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for (aoi_id, year), candidates in sorted(grouped.items()):
        aoi = config["aois"][aoi_id]
        transform, width, height, aoi_mask = target_grid(
            aoi["bounds_epsg4326"], dst_crs,
            float(policy["target_grid_resolution_m"]),
        )
        total = int(aoi_mask.sum())
        def evaluate(row):
            footprint, valid = scene_masks(
                row["SCL_href"], token, dst_crs, transform, width, height,
                aoi_mask, excluded,
            )
            footprint_count, valid_count = int(footprint.sum()), int(valid.sum())
            footprint_pct = 100.0 * footprint_count / total
            valid_pct = 100.0 * valid_count / total
            cloud = float(row["cloud_cover_pct"])
            score = (float(policy["coverage_weight"]) * valid_pct
                     + float(policy["catalog_cloud_weight"]) * (100.0 - cloud))
            return {
                **row, "_valid_mask": valid, "aoi_pixel_count": total,
                "scene_footprint_pixel_count": footprint_count,
                "valid_aoi_pixel_count": valid_count,
                "footprint_coverage_pct": round(footprint_pct, 4),
                "valid_coverage_pct": round(valid_pct, 4),
                "invalid_coverage_pct": round(100.0 - valid_pct, 4),
                "ranking_score": round(score, 4),
                "meets_minimum_valid_coverage": valid_pct >= minimum,
                "selected_for_composite": False, "selection_order": "",
                "incremental_valid_pixels": 0,
                "cumulative_valid_coverage_pct": 0.0,
                "status": "usable" if valid_pct >= minimum else "rejected",
                "reason": "meets_scl_coverage_threshold" if valid_pct >= minimum
                          else "below_scl_coverage_threshold",
            }
        # Remote cloud-optimized GeoTIFF window reads are independent. A small
        # worker pool keeps validation practical without downloading SCL files.
        with ThreadPoolExecutor(max_workers=min(4, len(candidates))) as executor:
            evaluated = list(executor.map(evaluate, candidates))
        evaluated.sort(key=lambda r: (-r["ranking_score"], -r["valid_coverage_pct"],
                                     float(r["cloud_cover_pct"]), r["acquisition_datetime"]))
        for index, row in enumerate(evaluated, 1):
            row["rank"] = index

        union = np.zeros_like(aoi_mask, dtype=bool)
        remaining = evaluated.copy()
        selected = []
        max_scenes = int(policy["maximum_scenes_per_composite"])
        min_scenes = int(policy["minimum_scenes_per_composite"])
        while remaining and len(selected) < max_scenes:
            best = max(remaining, key=lambda r: (
                int((r["_valid_mask"] & ~union).sum()), r["ranking_score"]
            ))
            gain = int((best["_valid_mask"] & ~union).sum())
            coverage = 100.0 * int(union.sum()) / total
            if gain == 0 or (coverage >= minimum and len(selected) >= min_scenes):
                break
            union |= best["_valid_mask"]
            remaining.remove(best)
            best["selected_for_composite"] = True
            best["selection_order"] = len(selected) + 1
            best["incremental_valid_pixels"] = gain
            best["cumulative_valid_coverage_pct"] = round(100.0 * int(union.sum()) / total, 4)
            selected.append(best)

        final_coverage = 100.0 * int(union.sum()) / total
        status = "passed" if final_coverage >= minimum else "failed"
        summaries.append({
            "aoi_id": aoi_id, "aoi_role": aoi["role"], "target_year": year,
            "candidate_count": len(evaluated),
            "individually_usable_count": sum(bool(r["meets_minimum_valid_coverage"]) for r in evaluated),
            "selected_count": len(selected),
            "selected_item_ids": [r["item_id"] for r in selected],
            "combined_valid_coverage_pct": round(final_coverage, 4),
            "minimum_required_pct": minimum, "status": status,
        })
        for row in evaluated:
            clean = {k: v for k, v in row.items() if k != "_valid_mask"}
            ranked_output.append(clean)
            if row["selected_for_composite"]:
                selected_output.append(clean)
    return ranked_output, selected_output, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/geospatial.yaml"))
    parser.add_argument("--input-manifest", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    input_path = args.input_manifest or Path(outputs["sentinel2_manifest"])
    rows = read_rows(input_path)
    token = sas_token(config["sentinel2"]["catalog_url"], config["sentinel2"]["collection"])
    ranked, selected, summaries = rank_and_select(rows, config, token)
    write_rows(Path(outputs["sentinel2_ranked_manifest"]), RANKED_FIELDS, ranked)
    write_rows(Path(outputs["sentinel2_composite_inputs"]), RANKED_FIELDS, selected)

    expected_summary_count = len(config["aois"]) * 2

    result = {
        "task": "3.3-rank-scenes-and-validate-scl-coverage",
        "completed_at_utc": utc_now(),
        "source_manifest": str(input_path),
        "excluded_scl_classes": config["sentinel2"]["excluded_scl_classes"],
        "minimum_valid_coverage_pct": config["sentinel2"]["minimum_valid_coverage_pct"],
        "expected_aoi_year_count": expected_summary_count,
        "actual_aoi_year_count": len(summaries),
        "status": (
            "passed"
            if len(summaries) == expected_summary_count
            and all(s["status"] == "passed" for s in summaries)
            else "failed"
        ),
        "aoi_year_results": summaries,
    }

    Path(outputs["sentinel2_coverage_summary"]).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for s in summaries:
        print(f"{s['aoi_id']:24} {s['target_year']}: {s['selected_count']} selected, "
              f"{s['combined_valid_coverage_pct']:.2f}% coverage [{s['status']}]")
    print(f"Task 3.3 validation: {result['status'].upper()}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())