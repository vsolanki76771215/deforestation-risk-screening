from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Task 3.6 label rasters")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with Path(args.manifest).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    expected_aois = set(config["aois"])
    errors: list[str] = []
    summary = {"task": "3.6", "status": "FAILED", "aois": {}, "errors": errors}
    nodata = int(config.get("labels", {}).get("output_nodata", 255))
    if len(rows) != len(expected_aois):
        errors.append(f"Manifest has {len(rows)} rows; expected {len(expected_aois)}")
    actual_aois = {row.get("aoi", "") for row in rows}
    if actual_aois != expected_aois:
        errors.append(f"Manifest AOIs differ; missing={sorted(expected_aois-actual_aois)}, extra={sorted(actual_aois-expected_aois)}")

    for row in rows:
        aoi = row.get("aoi", "")
        path = Path(row.get("output_path", ""))
        aoi_errors: list[str] = []
        if not path.exists():
            aoi_errors.append(f"Missing label raster: {path}")
        else:
            try:
                with rasterio.open(path) as dataset:
                    values = dataset.read(1)
                    valid = values != nodata
                    unique = sorted(int(v) for v in np.unique(values[valid]))
                    if dataset.count != 1: aoi_errors.append(f"band count {dataset.count}, expected 1")
                    if dataset.dtypes[0] != "uint8": aoi_errors.append(f"dtype {dataset.dtypes[0]}, expected uint8")
                    if dataset.nodata != nodata: aoi_errors.append(f"nodata {dataset.nodata}, expected {nodata}")
                    if dataset.crs is None: aoi_errors.append("CRS is undefined")
                    if not set(unique).issubset({0, 1}): aoi_errors.append(f"invalid label values: {unique}")
                    if not valid.any(): aoi_errors.append("no eligible label pixels")
                    positive = int((values == 1).sum())
                    negative = int((values == 0).sum())
                    expected_positive = int(row.get("positive_pixels", -1))
                    expected_negative = int(row.get("negative_pixels", -1))
                    if positive != expected_positive: aoi_errors.append(f"positive count {positive} != manifest {expected_positive}")
                    if negative != expected_negative: aoi_errors.append(f"negative count {negative} != manifest {expected_negative}")
                    summary["aois"][aoi] = {
                        "path": str(path), "crs": str(dataset.crs), "width": dataset.width,
                        "height": dataset.height, "values": unique, "positive_pixels": positive,
                        "negative_pixels": negative, "nodata_pixels": int((values == nodata).sum()),
                        "status": "PASSED" if not aoi_errors else "FAILED",
                    }
            except Exception as exc:
                aoi_errors.append(f"cannot read raster: {exc}")
        errors.extend(f"{aoi}: {message}" for message in aoi_errors)
        if aoi not in summary["aois"]:
            summary["aois"][aoi] = {"path": str(path), "status": "FAILED"}
        print(f"{aoi}: {'PASSED' if not aoi_errors else 'FAILED'}")
        for message in aoi_errors:
            print(f"  [FAILED] {message}")

    summary["status"] = "PASSED" if not errors else "FAILED"
    summary_path = Path(config["outputs"]["label_validation_summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"\nValidation summary: {summary_path}")
    print(f"Task 3.6 validation: {summary['status']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())