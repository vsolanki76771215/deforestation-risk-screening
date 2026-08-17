"""Build aligned, model-ready Sentinel-2 feature stacks for Task 3.7.

The output for each AOI is an 11-band float32 GeoTIFF plus a uint8 model-valid
mask.  A CSV manifest records lineage, coverage, spatial metadata, and SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
import yaml
from affine import Affine
from rasterio.windows import Window


DEFAULT_FEATURE_NAMES = [
    "blue_2018", "green_2018", "red_2018", "nir_2018",
    "blue_2022", "green_2022", "red_2022", "nir_2022",
    "ndvi_2018", "ndvi_2022", "ndvi_change",
]
DEFAULT_SOURCE_BANDS = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08"}
BAND_ALIASES = {
    "B02": {"b02", "b2", "blue", "b02_10m", "blue_10m"},
    "B03": {"b03", "b3", "green", "b03_10m", "green_10m"},
    "B04": {"b04", "b4", "red", "b04_10m", "red_10m"},
    "B08": {"b08", "b8", "nir", "b08_10m", "nir_10m"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/geospatial.yaml")
    parser.add_argument("--composite-manifest", default="data/manifests/sentinel2_composites.csv")
    parser.add_argument("--label-manifest", default="data/manifests/label_rasters.csv")
    parser.add_argument("--output-root", default="data/processed/feature_stacks")
    parser.add_argument("--output-manifest", default="data/manifests/model_features.csv")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return data


def resolve_existing(raw: Any, manifest_path: Path) -> Path:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)) or not str(raw).strip():
        raise ValueError(f"Manifest {manifest_path} contains an empty raster path")
    candidate = Path(str(raw).replace("\\", "/"))
    options = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate, manifest_path.parent / candidate]
    for option in options:
        if option.exists():
            return option.resolve()
    raise FileNotFoundError(f"Referenced file does not exist: {raw}")


def first_column(frame: pd.DataFrame, names: Iterable[str], purpose: str) -> str:
    normalized_columns = {str(column).strip().lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in normalized_columns:
            return normalized_columns[name.lower()]
    raise ValueError(f"Missing {purpose} column. Expected one of: {', '.join(names)}")


def normalize_description(value: str | None) -> str:
    return "" if value is None else value.strip().lower().replace("-", "_").replace(" ", "_")


def band_mapping(dataset: rasterio.io.DatasetReader, source_bands: dict[str, str], config: dict[str, Any]) -> dict[str, int]:
    normalized = [normalize_description(item) for item in dataset.descriptions]
    explicit = config.get("source_band_indices", {}) or config.get("band_indices", {})
    result: dict[str, int] = {}
    for color, code in source_bands.items():
        aliases = set(BAND_ALIASES.get(str(code).upper(), set()))
        aliases.update({normalize_description(color), normalize_description(str(code))})
        matches = [i + 1 for i, description in enumerate(normalized) if description in aliases]
        if len(matches) == 1:
            result[color] = matches[0]
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous descriptions for {code} in {dataset.name}: bands {matches}")
        elif color in explicit or code in explicit:
            index = int(explicit.get(color, explicit.get(code)))
            if not 1 <= index <= dataset.count:
                raise ValueError(f"Configured band index {index} for {code} is outside 1..{dataset.count}")
            result[color] = index
        else:
            raise ValueError(
                f"Cannot locate {code} in {dataset.name}; descriptions={dataset.descriptions}. "
                "Add features.source_band_indices to geospatial.yaml."
            )
    return result


def transforms_equal(left: Affine, right: Affine, tolerance: float = 1e-6) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def verify_alignment(reference: rasterio.io.DatasetReader, other: rasterio.io.DatasetReader, name: str) -> None:
    problems: list[str] = []
    if reference.crs != other.crs:
        problems.append(f"CRS {reference.crs} != {other.crs}")
    if (reference.width, reference.height) != (other.width, other.height):
        problems.append(f"shape {(reference.width, reference.height)} != {(other.width, other.height)}")
    if not transforms_equal(reference.transform, other.transform):
        problems.append(f"transform {reference.transform} != {other.transform}")
    if any(abs(a - b) > 1e-6 for a, b in zip(reference.res, other.res)):
        problems.append(f"resolution {reference.res} != {other.res}")
    if problems:
        raise ValueError(f"Grid alignment failed for {name}: " + "; ".join(problems))


def windows(width: int, height: int, block_size: int = 512) -> Iterable[Window]:
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            yield Window(col, row, min(block_size, width - col), min(block_size, height - row))


def should_scale(dataset: rasterio.io.DatasetReader, indices: Iterable[int], scale: float) -> bool:
    if scale == 1.0:
        return False
    if any(np.issubdtype(np.dtype(dataset.dtypes[index - 1]), np.integer) for index in indices):
        return True
    # Sample a centered window when floating-point storage does not reveal units.
    size = min(512, dataset.width, dataset.height)
    sample_window = Window((dataset.width - size) // 2, (dataset.height - size) // 2, size, size)
    maxima = []
    for index in indices:
        sample = dataset.read(index, window=sample_window, masked=True)
        if sample.count():
            maxima.append(float(np.nanpercentile(sample.compressed(), 99)))
    return bool(maxima and max(maxima) > 2.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("/", "\\")
    except ValueError:
        return str(path)


def build_one(
    aoi_id: str,
    role: str,
    baseline_path: Path,
    comparison_path: Path,
    label_path: Path,
    baseline_mask_path: Path | None,
    comparison_mask_path: Path | None,
    output_root: Path,
    feature_config: dict[str, Any],
) -> dict[str, Any]:
    baseline_year = int(feature_config.get("baseline_year", 2018))
    comparison_year = int(feature_config.get("comparison_year", 2022))
    source_bands = {**DEFAULT_SOURCE_BANDS, **(feature_config.get("source_bands", {}) or {})}
    feature_names = feature_config.get("output_bands", DEFAULT_FEATURE_NAMES)
    if feature_names != DEFAULT_FEATURE_NAMES:
        raise ValueError(f"features.output_bands must exactly equal the Task 3.7 contract: {DEFAULT_FEATURE_NAMES}")
    scale = float(feature_config.get("reflectance_scale", 10000.0))
    epsilon = float(feature_config.get("ndvi_epsilon", 1e-6))
    nodata = float(feature_config.get("output_nodata", -9999.0))
    minimum_coverage = float(feature_config.get("minimum_valid_coverage_pct", 80.0))
    expected_crs = str(feature_config.get("expected_crs", "EPSG:32719"))
    expected_resolution = float(feature_config.get("resolution_m", 10.0))

    destination = output_root / aoi_id
    destination.mkdir(parents=True, exist_ok=True)
    feature_path = destination / "model_features.tif"
    valid_mask_path = destination / "model_valid_mask.tif"

    with rasterio.open(baseline_path) as base, rasterio.open(comparison_path) as comp, rasterio.open(label_path) as label:
        verify_alignment(base, comp, f"{aoi_id} comparison composite")
        verify_alignment(base, label, f"{aoi_id} label raster")
        if base.crs is None or base.crs.to_string().upper() != expected_crs.upper():
            raise ValueError(f"{aoi_id}: expected {expected_crs}, found {base.crs}")
        if any(abs(value - expected_resolution) > 1e-6 for value in base.res):
            raise ValueError(f"{aoi_id}: expected {expected_resolution} m pixels, found {base.res}")

        base_map = band_mapping(base, source_bands, feature_config)
        comp_map = band_mapping(comp, source_bands, feature_config)
        base_scale = scale if should_scale(base, base_map.values(), scale) else 1.0
        comp_scale = scale if should_scale(comp, comp_map.values(), scale) else 1.0

        mask_contexts = []
        try:
            base_external = rasterio.open(baseline_mask_path) if baseline_mask_path else None
            comp_external = rasterio.open(comparison_mask_path) if comparison_mask_path else None
            mask_contexts.extend(item for item in (base_external, comp_external) if item is not None)
            if base_external:
                verify_alignment(base, base_external, f"{aoi_id} baseline valid mask")
            if comp_external:
                verify_alignment(base, comp_external, f"{aoi_id} comparison valid mask")

            feature_profile = base.profile.copy()
            feature_profile.update(driver="GTiff", count=11, dtype="float32", nodata=nodata,
                                   compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                                   BIGTIFF="IF_SAFER", predictor=3)
            mask_profile = base.profile.copy()
            mask_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0,
                                compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                                BIGTIFF="IF_SAFER")

            valid_count = 0
            label_eligible_count = 0
            total_count = base.width * base.height
            with rasterio.open(feature_path, "w", **feature_profile) as feature_dst, rasterio.open(valid_mask_path, "w", **mask_profile) as mask_dst:
                for band_number, name in enumerate(feature_names, start=1):
                    feature_dst.set_band_description(band_number, name)
                mask_dst.set_band_description(1, "model_valid")

                for window in windows(base.width, base.height):
                    base_arrays = {key: base.read(index, window=window, masked=True).astype("float32") / base_scale for key, index in base_map.items()}
                    comp_arrays = {key: comp.read(index, window=window, masked=True).astype("float32") / comp_scale for key, index in comp_map.items()}
                    # Read uint8 labels and their validity mask separately. This
                    # avoids assigning the feature nodata value (-9999) to an
                    # unsigned integer masked array.
                    labels = label.read(1, window=window, masked=False)
                    label_valid_mask = label.read_masks(1, window=window) != 0
                    label_eligible = label_valid_mask & np.isin(labels, (0, 1))
                    label_eligible_count += int(label_eligible.sum())
                    valid = label_eligible.copy()
                    for array in [*base_arrays.values(), *comp_arrays.values()]:
                        valid &= ~np.ma.getmaskarray(array) & np.isfinite(array.filled(np.nan))
                    if base_external:
                        valid &= base_external.read(1, window=window) != 0
                    if comp_external:
                        valid &= comp_external.read(1, window=window) != 0

                    b18 = {key: value.filled(np.nan) for key, value in base_arrays.items()}
                    b22 = {key: value.filled(np.nan) for key, value in comp_arrays.items()}
                    denominator18 = b18["nir"] + b18["red"]
                    denominator22 = b22["nir"] + b22["red"]
                    valid &= np.abs(denominator18) > epsilon
                    valid &= np.abs(denominator22) > epsilon
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ndvi18 = (b18["nir"] - b18["red"]) / (denominator18 + np.copysign(epsilon, denominator18))
                        ndvi22 = (b22["nir"] - b22["red"]) / (denominator22 + np.copysign(epsilon, denominator22))
                    change = ndvi22 - ndvi18
                    valid &= np.isfinite(ndvi18) & np.isfinite(ndvi22) & np.isfinite(change)
                    valid &= (ndvi18 >= -1.0001) & (ndvi18 <= 1.0001)
                    valid &= (ndvi22 >= -1.0001) & (ndvi22 <= 1.0001)
                    valid &= (change >= -2.0002) & (change <= 2.0002)

                    stack = np.stack([
                        b18["blue"], b18["green"], b18["red"], b18["nir"],
                        b22["blue"], b22["green"], b22["red"], b22["nir"],
                        ndvi18, ndvi22, change,
                    ]).astype("float32")
                    stack[:, ~valid] = nodata
                    feature_dst.write(stack, window=window)
                    mask_dst.write(valid.astype("uint8"), 1, window=window)
                    valid_count += int(valid.sum())
        finally:
            for item in mask_contexts:
                item.close()

    # Pass/fail is based on the pixels eligible for modeling according to the
    # Task 3.6 label contract. Pixels outside that label universe are
    # intentionally nodata and must not reduce model-ready coverage.
    model_ready_coverage = (
        100.0 * valid_count / label_eligible_count
        if label_eligible_count else 0.0
    )
    overall_aoi_coverage = 100.0 * valid_count / total_count if total_count else 0.0
    status = "passed" if model_ready_coverage >= minimum_coverage else "failed"
    return {
        "aoi_id": aoi_id, "aoi_role": role,
        "baseline_year": baseline_year, "comparison_year": comparison_year,
        "baseline_composite_path": relative_display(baseline_path),
        "comparison_composite_path": relative_display(comparison_path),
        "label_path": relative_display(label_path),
        "feature_stack_path": relative_display(feature_path),
        "model_valid_mask_path": relative_display(valid_mask_path),
        "feature_count": 11, "feature_names": "|".join(feature_names),
        "source_band_mapping": json.dumps({"baseline": base_map, "comparison": comp_map}, sort_keys=True),
        "baseline_reflectance_scale_applied": base_scale,
        "comparison_reflectance_scale_applied": comp_scale,
        "width": base.width, "height": base.height, "crs": base.crs.to_string(),
        "resolution_m": abs(base.res[0]), "nodata": nodata,
        "valid_pixel_count": valid_count,
        "label_eligible_pixel_count": label_eligible_count,
        "aoi_pixel_count": total_count,
        "valid_coverage_pct": round(model_ready_coverage, 4),
        "overall_aoi_coverage_pct": round(overall_aoi_coverage, 4),
        "sha256": sha256(feature_path), "status": status,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    composite_manifest_path = Path(args.composite_manifest)
    label_manifest_path = Path(args.label_manifest)
    output_root = Path(args.output_root)
    output_manifest = Path(args.output_manifest)

    config = load_yaml(config_path)
    feature_config = config.get("features", {})
    if not feature_config:
        raise ValueError("Missing required top-level 'features' section in geospatial.yaml")
    baseline_year = int(feature_config.get("baseline_year", 2018))
    comparison_year = int(feature_config.get("comparison_year", 2022))

    composites = pd.read_csv(composite_manifest_path)
    labels = pd.read_csv(label_manifest_path)
    aoi_col = first_column(composites, ("aoi_id", "aoi"), "composite AOI")
    year_col = first_column(composites, ("target_year", "year"), "composite year")
    composite_path_col = first_column(composites, ("composite_path", "raster_path", "path"), "composite path")
    valid_mask_col = next((item for item in ("valid_mask_path", "composite_valid_mask_path") if item in composites.columns), None)
    label_aoi_col = first_column(labels, ("aoi_id", "aoi"), "label AOI")
    label_path_col = first_column(
        labels,
        (
            "label_path",
            "label_raster_path",
            "binary_label_path",
            "loss_binary_path",
            "output_path",
            "raster_path",
            "path",
        ),
        "label path",
    )
    role_col = next((item for item in ("aoi_role", "role") if item in composites.columns), None)
    label_role_col = next((item for item in ("aoi_role", "role") if item in labels.columns), None)

    results: list[dict[str, Any]] = []
    for aoi_id in sorted(labels[label_aoi_col].astype(str).unique()):
        label_rows = labels[labels[label_aoi_col].astype(str) == aoi_id]
        if len(label_rows) != 1:
            raise ValueError(f"Expected exactly one label row for {aoi_id}, found {len(label_rows)}")
        selected: dict[int, pd.Series] = {}
        for year in (baseline_year, comparison_year):
            rows = composites[(composites[aoi_col].astype(str) == aoi_id) & (pd.to_numeric(composites[year_col]) == year)]
            if len(rows) != 1:
                raise ValueError(f"Expected exactly one composite for {aoi_id}/{year}, found {len(rows)}")
            selected[year] = rows.iloc[0]
        label_row = label_rows.iloc[0]
        role = str(selected[baseline_year][role_col]) if role_col else (
            str(label_row[label_role_col]) if label_role_col else
            ("geographic_holdout" if aoi_id == "tambopata_test_area" else "dataset_development")
        )
        get_mask = lambda row: resolve_existing(row[valid_mask_col], composite_manifest_path) if valid_mask_col and pd.notna(row[valid_mask_col]) else None
        print(f"{aoi_id}")
        result = build_one(
            aoi_id, role,
            resolve_existing(selected[baseline_year][composite_path_col], composite_manifest_path),
            resolve_existing(selected[comparison_year][composite_path_col], composite_manifest_path),
            resolve_existing(label_row[label_path_col], label_manifest_path),
            get_mask(selected[baseline_year]), get_mask(selected[comparison_year]),
            output_root, feature_config,
        )
        results.append(result)
        print("  Alignment: passed")
        print(f"  Features: {result['feature_count']}")
        print(f"  Model-ready coverage of eligible labels: {result['valid_coverage_pct']:.4f}%")
        print(f"  Overall AOI coverage: {result['overall_aoi_coverage_pct']:.4f}%")
        print(f"  Status: {result['status']}")

    if not results:
        raise ValueError("No AOIs were found in the label manifest")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_manifest, index=False)
    print(f"\nFeature manifest: {relative_display(output_manifest)}")
    failed = [row["aoi_id"] for row in results if row["status"] != "passed"]
    if failed:
        print(f"Task 3.7 feature generation: FAILED ({', '.join(failed)})", file=sys.stderr)
        return 1
    print("Task 3.7 feature generation: COMPLETED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)