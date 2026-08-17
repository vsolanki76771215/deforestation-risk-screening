#!/usr/bin/env python3
"""Validate Task 3.7 model-ready Sentinel-2 feature stacks.

The validator checks the feature contract, raster alignment, valid-mask
semantics, numeric ranges, coverage, hashes, and geographic-holdout role. It
processes rasters block-by-block to keep memory use bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
import yaml


DEFAULT_FEATURES = [
    "blue_2018", "green_2018", "red_2018", "nir_2018",
    "blue_2022", "green_2022", "red_2022", "nir_2022",
    "ndvi_2018", "ndvi_2022", "ndvi_change",
]
REFLECTANCE_INDICES = tuple(range(8))
NDVI_INDICES = (8, 9)
CHANGE_INDEX = 10
TARGET_NAMES = {
    "label", "loss_binary", "loss_fraction", "treecover", "treecover2000",
    "lossyear", "hansen", "target", "class",
}


class ValidationError(RuntimeError):
    pass


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        values64 = values.astype(np.float64, copy=False)
        self.count += int(values64.size)
        self.total += float(values64.sum(dtype=np.float64))
        self.total_sq += float(np.square(values64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values64.min()))
        self.maximum = max(self.maximum, float(values64.max()))

    def as_dict(self) -> dict[str, Any]:
        if not self.count:
            return {"minimum": None, "maximum": None, "mean": None,
                    "standard_deviation": None, "valid_pixel_count": 0}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "valid_pixel_count": self.count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/geospatial.yaml")
    parser.add_argument("--feature-manifest", default="data/manifests/model_features.csv")
    parser.add_argument(
        "--output-summary",
        default="data/manifests/model_feature_validation_summary.json",
    )
    parser.add_argument("--minimum-coverage-pct", type=float, default=None)
    return parser.parse_args()


def norm_headers(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame


def first_column(frame: pd.DataFrame, names: Iterable[str], purpose: str) -> str:
    matches = [name for name in names if name in frame.columns]
    if not matches:
        raise ValidationError(
            f"Missing {purpose} column. Expected one of: {', '.join(names)}"
        )
    return matches[0]


def resolve_path(raw: Any, project_root: Path) -> Path:
    text = str(raw).strip().replace("\\", "/")
    path = Path(text)
    return path if path.is_absolute() else project_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transforms_match(left: rasterio.Affine, right: rasterio.Affine,
                     tolerance: float = 1e-6) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=tolerance))


def grids_match(left: rasterio.io.DatasetReader,
                right: rasterio.io.DatasetReader) -> bool:
    return (
        left.crs == right.crs
        and left.width == right.width
        and left.height == right.height
        and transforms_match(left.transform, right.transform)
        and np.allclose(left.res, right.res, rtol=0.0, atol=1e-6)
        and np.allclose(left.bounds, right.bounds, rtol=0.0, atol=1e-6)
    )


def config_contract(config_path: Path) -> tuple[list[str], str, float, float, float]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    features = config.get("features", {}) or {}
    names = list(features.get("output_bands", DEFAULT_FEATURES))
    crs = str(features.get("expected_crs", config.get("analysis_crs", "EPSG:32719")))
    resolution = float(features.get("resolution_m", 10.0))
    nodata = float(features.get("output_nodata", -9999.0))
    minimum = float(features.get("minimum_valid_coverage_pct", 80.0))
    return names, crs, resolution, nodata, minimum


def add_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def validate_aoi(
    row: pd.Series,
    columns: dict[str, str],
    project_root: Path,
    expected_names: list[str],
    expected_crs: str,
    expected_resolution: float,
    expected_nodata: float,
    minimum_coverage: float,
) -> dict[str, Any]:
    aoi = str(row[columns["aoi"]]).strip()
    role = str(row[columns["role"]]).strip() if columns.get("role") else ""
    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {"aoi_id": aoi, "aoi_role": role}

    feature_path = resolve_path(row[columns["feature"]], project_root)
    mask_path = resolve_path(row[columns["mask"]], project_root)
    label_path = resolve_path(row[columns["label"]], project_root)
    result.update({
        "feature_stack_path": str(row[columns["feature"]]),
        "model_valid_mask_path": str(row[columns["mask"]]),
        "label_path": str(row[columns["label"]]),
    })

    for description, path in (("feature stack", feature_path),
                              ("model-valid mask", mask_path),
                              ("label raster", label_path)):
        if not path.is_file():
            add_error(errors, f"Missing {description}: {path}")
    if errors:
        result.update(status="failed", errors=errors, warnings=warnings,
                      feature_statistics={})
        return result

    expected_crs_obj = rasterio.crs.CRS.from_string(expected_crs)
    stats = [RunningStats() for _ in expected_names]
    total_pixels = 0
    label_eligible = 0
    model_valid = 0
    invalid_feature_values = 0
    nonfinite_valid_values = 0
    inconsistent_mask_values = 0
    invalid_valid_labels = 0
    range_violations = {"ndvi": 0, "ndvi_change": 0}

    try:
        with rasterio.open(feature_path) as feature, rasterio.open(mask_path) as mask, \
                rasterio.open(label_path) as label:
            names = [str(x).strip() if x is not None else "" for x in feature.descriptions]
            result.update({
                "width": feature.width,
                "height": feature.height,
                "crs": str(feature.crs),
                "resolution_m": float(abs(feature.res[0])),
                "feature_count": feature.count,
                "feature_names": names,
                "nodata": feature.nodata,
            })

            if feature.count != len(expected_names):
                add_error(errors, f"Expected {len(expected_names)} feature bands; found {feature.count}")
            if names != expected_names:
                add_error(errors, f"Feature names/order mismatch: {names}")
            lowered = [name.lower() for name in names]
            leaked = [name for name in lowered if any(token in name for token in TARGET_NAMES)]
            if leaked:
                add_error(errors, f"Target-derived feature name(s): {', '.join(leaked)}")
            if feature.crs != expected_crs_obj:
                add_error(errors, f"Expected CRS {expected_crs}; found {feature.crs}")
            if not np.allclose(np.abs(feature.res), [expected_resolution, expected_resolution],
                               rtol=0.0, atol=1e-6):
                add_error(errors, f"Expected {expected_resolution:g} m resolution; found {feature.res}")
            if any(dtype != "float32" for dtype in feature.dtypes):
                add_error(errors, f"All feature bands must be float32; found {feature.dtypes}")
            if feature.nodata is None or not math.isclose(float(feature.nodata), expected_nodata,
                                                           rel_tol=0.0, abs_tol=1e-6):
                add_error(errors, f"Expected feature nodata {expected_nodata}; found {feature.nodata}")
            if mask.count != 1 or mask.dtypes[0] != "uint8":
                add_error(errors, f"Model-valid mask must be one uint8 band; found count={mask.count}, dtype={mask.dtypes[0]}")
            if not grids_match(feature, mask):
                add_error(errors, "Feature stack and model-valid mask grids do not match")
            if not grids_match(feature, label):
                add_error(errors, "Feature stack and label raster grids do not match")

            if not errors:
                for _, window in feature.block_windows(1):
                    values = feature.read(window=window)
                    valid_mask_raw = mask.read(1, window=window)
                    label_values = label.read(1, window=window, masked=False)
                    label_data_mask = label.read_masks(1, window=window) != 0

                    total_pixels += int(valid_mask_raw.size)
                    unexpected = ~np.isin(valid_mask_raw, (0, 1))
                    inconsistent_mask_values += int(unexpected.sum())
                    valid = valid_mask_raw == 1
                    eligible = label_data_mask & np.isin(label_values, (0, 1))
                    label_eligible += int(eligible.sum())
                    model_valid += int(valid.sum())
                    invalid_valid_labels += int((valid & ~eligible).sum())

                    finite = np.isfinite(values)
                    nonfinite_valid_values += int((~finite[:, valid]).sum()) if valid.any() else 0
                    invalid = ~valid
                    if invalid.any():
                        expected_invalid = np.isclose(values[:, invalid], expected_nodata,
                                                      rtol=0.0, atol=1e-6)
                        invalid_feature_values += int((~expected_invalid).sum())

                    if valid.any():
                        for index in range(min(len(stats), values.shape[0])):
                            band_values = values[index, valid]
                            finite_values = band_values[np.isfinite(band_values)]
                            stats[index].update(finite_values)
                        for index in NDVI_INDICES:
                            band = values[index, valid]
                            range_violations["ndvi"] += int(((band < -1.00001) | (band > 1.00001)).sum())
                        change = values[CHANGE_INDEX, valid]
                        range_violations["ndvi_change"] += int(((change < -2.00001) | (change > 2.00001)).sum())

    except Exception as exc:
        add_error(errors, f"Raster validation failed: {exc}")

    if inconsistent_mask_values:
        add_error(errors, f"Model-valid mask contains {inconsistent_mask_values} values other than 0 or 1")
    if nonfinite_valid_values:
        add_error(errors, f"Found {nonfinite_valid_values} NaN/infinite values at model-valid pixels")
    if invalid_feature_values:
        add_error(errors, f"Found {invalid_feature_values} invalid-pixel feature values not equal to nodata")
    if invalid_valid_labels:
        add_error(errors, f"Found {invalid_valid_labels} model-valid pixels without an eligible binary label")
    if range_violations["ndvi"]:
        add_error(errors, f"Found {range_violations['ndvi']} NDVI values outside [-1, 1]")
    if range_violations["ndvi_change"]:
        add_error(errors, f"Found {range_violations['ndvi_change']} NDVI-change values outside [-2, 2]")

    eligible_coverage = 100.0 * model_valid / label_eligible if label_eligible else 0.0
    overall_coverage = 100.0 * model_valid / total_pixels if total_pixels else 0.0
    result.update({
        "aoi_pixel_count": total_pixels,
        "label_eligible_pixel_count": label_eligible,
        "valid_pixel_count": model_valid,
        "valid_coverage_pct": round(eligible_coverage, 4),
        "overall_aoi_coverage_pct": round(overall_coverage, 4),
        "feature_statistics": {
            name: stats[index].as_dict() for index, name in enumerate(expected_names)
        },
    })
    if not label_eligible:
        add_error(errors, "No label-eligible pixels found")
    elif eligible_coverage + 1e-9 < minimum_coverage:
        add_error(errors, f"Eligible-label coverage {eligible_coverage:.4f}% is below {minimum_coverage:.4f}%")

    if columns.get("valid_count"):
        manifest_count = int(float(row[columns["valid_count"]]))
        if manifest_count != model_valid:
            add_error(errors, f"Manifest valid_pixel_count={manifest_count}; calculated={model_valid}")
    if columns.get("eligible_count"):
        manifest_count = int(float(row[columns["eligible_count"]]))
        if manifest_count != label_eligible:
            add_error(errors, f"Manifest label_eligible_pixel_count={manifest_count}; calculated={label_eligible}")
    if columns.get("coverage"):
        manifest_coverage = float(row[columns["coverage"]])
        if not math.isclose(manifest_coverage, eligible_coverage, rel_tol=0.0, abs_tol=0.01):
            add_error(errors, f"Manifest valid_coverage_pct={manifest_coverage}; calculated={eligible_coverage:.4f}")

    actual_hash = sha256_file(feature_path)
    result["sha256"] = actual_hash
    if columns.get("sha256"):
        expected_hash = str(row[columns["sha256"]]).strip().lower()
        if expected_hash and expected_hash != actual_hash:
            add_error(errors, "Feature-stack SHA-256 does not match manifest")

    if aoi == "tambopata_test_area" and role != "geographic_holdout":
        add_error(errors, "tambopata_test_area must have role geographic_holdout")
    if aoi != "tambopata_test_area" and role and role != "dataset_development":
        warnings.append(f"Unexpected development AOI role: {role}")

    result.update(status="failed" if errors else "passed", errors=errors, warnings=warnings)
    return result


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    manifest_path = Path(args.feature_manifest).resolve()
    output_path = Path(args.output_summary).resolve()
    project_root = config_path.parent.parent

    try:
        if not config_path.is_file():
            raise ValidationError(f"Config not found: {config_path}")
        if not manifest_path.is_file():
            raise ValidationError(f"Feature manifest not found: {manifest_path}")
        expected_names, expected_crs, resolution, nodata, config_minimum = config_contract(config_path)
        minimum = args.minimum_coverage_pct if args.minimum_coverage_pct is not None else config_minimum
        if expected_names != DEFAULT_FEATURES:
            raise ValidationError(
                "Task 3.7 feature contract must exactly equal: " + "|".join(DEFAULT_FEATURES)
            )

        manifest = norm_headers(pd.read_csv(manifest_path))
        if manifest.empty:
            raise ValidationError("Feature manifest is empty")
        columns = {
            "aoi": first_column(manifest, ("aoi_id", "aoi"), "AOI"),
            "feature": first_column(manifest, ("feature_stack_path", "feature_path", "output_path"), "feature-stack path"),
            "mask": first_column(manifest, ("model_valid_mask_path", "valid_mask_path", "mask_path"), "model-valid-mask path"),
            "label": first_column(manifest, ("label_path", "label_raster_path", "binary_label_path"), "label path"),
            "role": next((x for x in ("aoi_role", "role") if x in manifest.columns), ""),
            "valid_count": next((x for x in ("valid_pixel_count",) if x in manifest.columns), ""),
            "eligible_count": next((x for x in ("label_eligible_pixel_count", "eligible_pixel_count") if x in manifest.columns), ""),
            "coverage": next((x for x in ("valid_coverage_pct",) if x in manifest.columns), ""),
            "sha256": next((x for x in ("sha256", "feature_sha256") if x in manifest.columns), ""),
        }
        duplicates = manifest[columns["aoi"]].astype(str).str.strip().duplicated(keep=False)
        if duplicates.any():
            values = sorted(manifest.loc[duplicates, columns["aoi"]].astype(str).unique())
            raise ValidationError(f"Duplicate AOI rows: {', '.join(values)}")

        results = [
            validate_aoi(row, columns, project_root, expected_names, expected_crs,
                         resolution, nodata, minimum)
            for _, row in manifest.iterrows()
        ]
        all_errors = [f"{r['aoi_id']}: {e}" for r in results for e in r["errors"]]
        all_warnings = [f"{r['aoi_id']}: {w}" for r in results for w in r["warnings"]]
        summary = {
            "task": "3.7",
            "status": "failed" if all_errors else "passed",
            "expected_feature_count": len(expected_names),
            "feature_names": expected_names,
            "minimum_valid_coverage_pct": minimum,
            "coverage_denominator": "label_eligible_pixels",
            "aoi_results": results,
            "errors": all_errors,
            "warnings": all_warnings,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        for result in results:
            print(f"{result['aoi_id']}: {result['status'].upper()}")
            if "valid_coverage_pct" in result:
                print(f"  Eligible-label coverage: {result['valid_coverage_pct']:.4f}%")
                print(f"  Overall AOI coverage: {result['overall_aoi_coverage_pct']:.4f}%")
            for error in result["errors"]:
                print(f"  ERROR: {error}")
            for warning in result["warnings"]:
                print(f"  WARNING: {warning}")
        print(f"\nValidation summary: {args.output_summary}")
        print(f"Task 3.7 validation: {summary['status'].upper()}")
        return 1 if all_errors else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())