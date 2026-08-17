from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml


LAYERS = ("treecover2000", "lossyear")


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"  [FAILED] {message}")


def same_transform(left: rasterio.Affine, right: rasterio.Affine) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=1e-12))


def inspect_raster(path: Path, configured_nodata: int) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        values = dataset.read(1)
        nodata = dataset.nodata
        effective_nodata = configured_nodata if nodata is None else nodata
        valid = values != effective_nodata
        valid_values = values[valid]

        return {
            "path": str(path),
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": dataset.crs,
            "crs_text": str(dataset.crs) if dataset.crs else None,
            "transform": dataset.transform,
            "transform_values": list(dataset.transform),
            "width": dataset.width,
            "height": dataset.height,
            "nodata": nodata,
            "valid_mask": valid,
            "values": values,
            "valid_pixels": int(valid.sum()),
            "total_pixels": int(values.size),
            "coverage_pct": float(valid.sum() / values.size * 100) if values.size else 0.0,
            "min_value": int(valid_values.min()) if valid_values.size else None,
            "max_value": int(valid_values.max()) if valid_values.size else None,
        }


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Task 3.5 Hansen GFC AOI clips")
    parser.add_argument("--config", required=True, help="Path to geospatial.yaml")
    parser.add_argument("--manifest", required=True, help="Path to hansen_clipped.csv")
    args = parser.parse_args()

    config_path = Path(args.config)
    manifest_path = Path(args.manifest)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    hansen = config["hansen"]
    outputs = config["outputs"]
    aoi_names = list(config["aois"])
    clipped_root = Path(outputs["hansen_clipped_root"])
    summary_path = Path(outputs["hansen_validation_summary"])

    expected_dtype = str(hansen.get("output_dtype", "uint8"))
    configured_nodata = int(hansen.get("output_nodata", 255))
    minimum_coverage = float(hansen.get("minimum_valid_coverage_pct", 95))
    forest_threshold = int(hansen["forest_threshold_treecover2000_pct"])
    positive_start, positive_end = map(int, hansen["positive_loss_years"])
    version = str(hansen["version"])
    expected_loss_max = 25 if "v1.13" in version else positive_end - 2000

    errors: list[str] = []
    manifest_rows = load_manifest(manifest_path)
    expected_pairs = {(aoi, layer) for aoi in aoi_names for layer in LAYERS}
    actual_pairs = {(row.get("aoi", ""), row.get("layer", "")) for row in manifest_rows}

    print(f"Hansen version: {version}")
    print(f"Manifest: {manifest_path}")

    if len(manifest_rows) != len(expected_pairs):
        fail(errors, f"Manifest has {len(manifest_rows)} rows; expected {len(expected_pairs)}")
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        if missing:
            fail(errors, f"Manifest is missing records: {missing}")
        if extra:
            fail(errors, f"Manifest contains unexpected records: {extra}")

    summary: dict[str, Any] = {
        "task": "3.5",
        "hansen_version": version,
        "status": "FAILED",
        "minimum_valid_coverage_pct": minimum_coverage,
        "forest_threshold_treecover2000_pct": forest_threshold,
        "positive_loss_years": [positive_start, positive_end],
        "aois": {},
        "errors": errors,
    }

    for aoi_name in aoi_names:
        print(f"\n{aoi_name}")
        rasters: dict[str, dict[str, Any]] = {}

        for layer in LAYERS:
            path = clipped_root / aoi_name / f"{layer}.tif"
            if not path.exists():
                fail(errors, f"Missing {path}")
                continue
            try:
                info = inspect_raster(path, configured_nodata)
                rasters[layer] = info
            except Exception as exc:
                fail(errors, f"Cannot read {path}: {exc}")
                continue

            layer_ok = True
            if info["count"] != 1:
                fail(errors, f"{layer}: band count is {info['count']}; expected 1")
                layer_ok = False
            if info["dtype"] != expected_dtype:
                fail(errors, f"{layer}: dtype is {info['dtype']}; expected {expected_dtype}")
                layer_ok = False
            if info["crs"] is None:
                fail(errors, f"{layer}: CRS is undefined")
                layer_ok = False
            if info["width"] <= 0 or info["height"] <= 0:
                fail(errors, f"{layer}: invalid dimensions {info['width']} x {info['height']}")
                layer_ok = False
            if info["valid_pixels"] == 0:
                fail(errors, f"{layer}: contains no valid pixels")
                layer_ok = False
            if info["coverage_pct"] < minimum_coverage:
                fail(errors, f"{layer}: coverage {info['coverage_pct']:.2f}% is below {minimum_coverage:.2f}%")
                layer_ok = False

            allowed_max = 100 if layer == "treecover2000" else expected_loss_max
            if info["min_value"] is not None and not (0 <= info["min_value"] <= allowed_max):
                fail(errors, f"{layer}: minimum value {info['min_value']} is outside 0-{allowed_max}")
                layer_ok = False
            if info["max_value"] is not None and not (0 <= info["max_value"] <= allowed_max):
                fail(errors, f"{layer}: maximum value {info['max_value']} is outside 0-{allowed_max}")
                layer_ok = False

            if layer_ok:
                print(
                    f"  {layer}: passed | {info['width']}x{info['height']} | "
                    f"range {info['min_value']}-{info['max_value']} | "
                    f"coverage {info['coverage_pct']:.2f}%"
                )

        aoi_summary: dict[str, Any] = {}
        if all(layer in rasters for layer in LAYERS):
            tree = rasters["treecover2000"]
            loss = rasters["lossyear"]
            aligned = (
                tree["crs"] == loss["crs"]
                and tree["width"] == loss["width"]
                and tree["height"] == loss["height"]
                and same_transform(tree["transform"], loss["transform"])
            )
            if not aligned:
                fail(errors, f"{aoi_name}: treecover2000 and lossyear grids are not aligned")
            else:
                print("  grid alignment: passed")

            if tree["values"].shape == loss["values"].shape:
                valid_mask = tree["valid_mask"] & loss["valid_mask"]
                annual_counts = {
                    str(year): int((valid_mask & (loss["values"] == year - 2000)).sum())
                    for year in range(positive_start, positive_end + 1)
                }
                positive_mask = (
                    valid_mask
                    & (tree["values"] >= forest_threshold)
                    & (loss["values"] >= positive_start - 2000)
                    & (loss["values"] <= positive_end - 2000)
                )
                for year, count in annual_counts.items():
                    print(f"  {year} loss pixels: {count:,}")
                print(f"  preliminary positive-label pixels: {int(positive_mask.sum()):,}")
            else:
                annual_counts = {}
                positive_mask = np.zeros((0,), dtype=bool)

            aoi_summary = {
                "grid_aligned": aligned,
                "crs": tree["crs_text"],
                "width": tree["width"],
                "height": tree["height"],
                "treecover2000_range": [tree["min_value"], tree["max_value"]],
                "lossyear_range": [loss["min_value"], loss["max_value"]],
                "treecover2000_coverage_pct": round(tree["coverage_pct"], 6),
                "lossyear_coverage_pct": round(loss["coverage_pct"], 6),
                "annual_loss_pixels": annual_counts,
                "preliminary_positive_label_pixels": int(positive_mask.sum()),
            }
        summary["aois"][aoi_name] = aoi_summary

    summary["status"] = "PASSED" if not errors else "FAILED"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"\nValidation summary: {summary_path}")
    print(f"Task 3.5 validation: {summary['status']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())