"""Build an inference-only 11-band Sentinel-2 feature stack without labels.

The feature order and calculations match ``src/features/build_feature_stacks.py``:
four reflectance bands for 2018, four for 2022, then NDVI for each year and
the 2022-minus-2018 NDVI change.  Pixels are valid only when both composites
and their valid masks are valid; no Hansen label data are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml


SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))
from features.build_feature_stacks import (  # noqa: E402
    DEFAULT_FEATURE_NAMES,
    band_mapping,
    should_scale,
    verify_alignment,
    windows,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(raw: str, manifest: Path) -> Path:
    candidate = Path(str(raw).replace("\\", "/"))
    for path in (candidate, Path.cwd() / candidate, manifest.parent / candidate):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Missing file referenced by {manifest}: {raw}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--composite-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    features = config["features"]
    aoi_id = next(iter(config["aois"]))
    baseline_year = int(features["baseline_year"])
    comparison_year = int(features["comparison_year"])
    nodata = float(features["output_nodata"])
    epsilon = float(features["ndvi_epsilon"])
    scale = float(features["reflectance_scale"])
    expected_crs = str(config["specification"]["analysis_crs"])

    with args.composite_manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = {
        int(row["target_year"]): row
        for row in rows
        if row["aoi_id"] == aoi_id and int(row["target_year"]) in (baseline_year, comparison_year)
    }
    if set(selected) != {baseline_year, comparison_year}:
        raise ValueError(f"Expected one composite for {aoi_id} in {baseline_year} and {comparison_year}")

    base_row, comp_row = selected[baseline_year], selected[comparison_year]
    base_path = locate(base_row["composite_path"], args.composite_manifest)
    comp_path = locate(comp_row["composite_path"], args.composite_manifest)
    base_mask_path = locate(base_row["valid_mask_path"], args.composite_manifest)
    comp_mask_path = locate(comp_row["valid_mask_path"], args.composite_manifest)
    destination = args.output_root / aoi_id
    destination.mkdir(parents=True, exist_ok=True)
    feature_path = destination / "model_features.tif"
    mask_path = destination / "model_valid_mask.tif"

    with rasterio.open(base_path) as base, rasterio.open(comp_path) as comp, \
            rasterio.open(base_mask_path) as base_mask, rasterio.open(comp_mask_path) as comp_mask:
        verify_alignment(base, comp, "comparison composite")
        verify_alignment(base, base_mask, "baseline valid mask")
        verify_alignment(base, comp_mask, "comparison valid mask")
        if base.crs is None or base.crs.to_string().upper() != expected_crs.upper():
            raise ValueError(f"Expected {expected_crs}; found {base.crs}")
        base_map = band_mapping(base, features["source_bands"], features)
        comp_map = band_mapping(comp, features["source_bands"], features)
        base_scale = scale if should_scale(base, base_map.values(), scale) else 1.0
        comp_scale = scale if should_scale(comp, comp_map.values(), scale) else 1.0
        profile = base.profile.copy()
        profile.update(driver="GTiff", count=11, dtype="float32", nodata=nodata,
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                       BIGTIFF="IF_SAFER", predictor=3)
        mask_profile = base.profile.copy()
        mask_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0,
                            compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                            BIGTIFF="IF_SAFER")
        valid_count = 0
        with rasterio.open(feature_path, "w", **profile) as dst, rasterio.open(mask_path, "w", **mask_profile) as valid_dst:
            for index, name in enumerate(DEFAULT_FEATURE_NAMES, 1):
                dst.set_band_description(index, name)
            valid_dst.set_band_description(1, "model_valid")
            for window in windows(base.width, base.height):
                b18 = {key: base.read(i, window=window, masked=True).astype("float32").filled(np.nan) / base_scale for key, i in base_map.items()}
                b22 = {key: comp.read(i, window=window, masked=True).astype("float32").filled(np.nan) / comp_scale for key, i in comp_map.items()}
                valid = (base_mask.read(1, window=window) != 0) & (comp_mask.read(1, window=window) != 0)
                for array in (*b18.values(), *b22.values()):
                    valid &= np.isfinite(array)
                d18, d22 = b18["nir"] + b18["red"], b22["nir"] + b22["red"]
                valid &= np.abs(d18) > epsilon
                valid &= np.abs(d22) > epsilon
                with np.errstate(divide="ignore", invalid="ignore"):
                    ndvi18 = (b18["nir"] - b18["red"]) / (d18 + np.copysign(epsilon, d18))
                    ndvi22 = (b22["nir"] - b22["red"]) / (d22 + np.copysign(epsilon, d22))
                change = ndvi22 - ndvi18
                valid &= np.isfinite(ndvi18) & np.isfinite(ndvi22) & np.isfinite(change)
                valid &= (ndvi18 >= -1.0001) & (ndvi18 <= 1.0001)
                valid &= (ndvi22 >= -1.0001) & (ndvi22 <= 1.0001)
                valid &= (change >= -2.0002) & (change <= 2.0002)
                stack = np.stack([b18["blue"], b18["green"], b18["red"], b18["nir"], b22["blue"], b22["green"], b22["red"], b22["nir"], ndvi18, ndvi22, change]).astype("float32")
                stack[:, ~valid] = nodata
                dst.write(stack, window=window)
                valid_dst.write(valid.astype("uint8"), 1, window=window)
                valid_count += int(valid.sum())

    total = base.width * base.height
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["aoi_id", "feature_stack_path", "model_valid_mask_path", "feature_names", "valid_pixel_count", "aoi_pixel_count", "overall_valid_coverage_pct", "sha256", "status"])
        writer.writeheader()
        writer.writerow({"aoi_id": aoi_id, "feature_stack_path": str(feature_path), "model_valid_mask_path": str(mask_path), "feature_names": "|".join(DEFAULT_FEATURE_NAMES), "valid_pixel_count": valid_count, "aoi_pixel_count": total, "overall_valid_coverage_pct": round(100.0 * valid_count / total, 4), "sha256": sha256(feature_path), "status": "passed"})
    print(f"Feature stack: {feature_path}")
    print(f"Valid-pixel coverage: {100.0 * valid_count / total:.4f}%")
    print(f"Manifest: {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())