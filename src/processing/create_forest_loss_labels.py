from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject


PATH_COLUMNS = ("output_path", "composite_path", "path", "raster_path")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_path(row: dict[str, str]) -> Path:
    for key in PATH_COLUMNS:
        if row.get(key):
            return Path(row[key])
    raise ValueError(f"Composite manifest row has no path column: {row}")


def find_composite(rows: list[dict[str, str]], aoi: str, year: int) -> Path:
    matches = [
        row for row in rows
        if (row.get("aoi_id") or row.get("aoi")) == aoi
        and str(row.get("target_year") or row.get("year") or "") == str(year)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Sentinel-2 composite for {aoi} {year}; found {len(matches)}"
        )
    path = row_path(matches[0])
    if not path.exists():
        raise FileNotFoundError(f"Composite not found: {path}")
    return path


def valid_imagery_mask(dataset: rasterio.DatasetReader) -> np.ndarray:
    masks = dataset.read_masks()
    valid = np.all(masks > 0, axis=0)
    values = dataset.read()
    valid &= np.all(np.isfinite(values), axis=0)
    if dataset.nodata is not None:
        valid &= np.all(values != dataset.nodata, axis=0)
    return valid


def aligned(left: rasterio.DatasetReader, right: rasterio.DatasetReader) -> bool:
    return (
        left.crs == right.crs
        and left.width == right.width
        and left.height == right.height
        and np.allclose(tuple(left.transform), tuple(right.transform), rtol=0, atol=1e-9)
    )


def warp_hansen(path: Path, reference: rasterio.DatasetReader, nodata: int) -> np.ndarray:
    destination = np.full((reference.height, reference.width), nodata, dtype="uint8")
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata if source.nodata is not None else nodata,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=nodata,
            resampling=Resampling.nearest,
        )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Task 3.6 forest-loss labels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--composite-manifest", required=True)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    outputs = config["outputs"]
    hansen = config["hansen"]
    labels = config.get("labels", {})
    rows = load_rows(Path(args.composite_manifest))
    baseline_year = int(config["sentinel2"]["baseline"]["year"])
    comparison_year = int(config["sentinel2"]["comparison"]["year"])
    forest_threshold = int(hansen["forest_threshold_treecover2000_pct"])
    prior_start, prior_end = map(int, hansen["prior_loss_exclusion_years"])
    positive_start, positive_end = map(int, hansen["positive_loss_years"])
    hansen_nodata = int(hansen.get("output_nodata", 255))
    output_nodata = int(labels.get("output_nodata", 255))
    label_root = Path(outputs["label_root"])
    hansen_root = Path(outputs["hansen_clipped_root"])
    manifest_path = Path(outputs["label_manifest"])
    manifest_rows: list[dict[str, object]] = []

    for aoi in config["aois"]:
        baseline_path = find_composite(rows, aoi, baseline_year)
        comparison_path = find_composite(rows, aoi, comparison_year)
        tree_path = hansen_root / aoi / "treecover2000.tif"
        loss_path = hansen_root / aoi / "lossyear.tif"
        if not tree_path.exists() or not loss_path.exists():
            raise FileNotFoundError(f"Missing validated Hansen rasters for {aoi}")

        with rasterio.open(baseline_path) as baseline, rasterio.open(comparison_path) as comparison:
            if not aligned(baseline, comparison):
                raise ValueError(f"Sentinel-2 baseline/comparison grids do not align for {aoi}")
            baseline_valid = valid_imagery_mask(baseline)
            comparison_valid = valid_imagery_mask(comparison)
            imagery_valid = baseline_valid & comparison_valid
            tree = warp_hansen(tree_path, baseline, hansen_nodata)
            loss = warp_hansen(loss_path, baseline, hansen_nodata)
            hansen_valid = (tree != hansen_nodata) & (loss != hansen_nodata)
            prior_loss = (loss >= prior_start - 2000) & (loss <= prior_end - 2000)
            eligible = hansen_valid & imagery_valid & (tree >= forest_threshold) & ~prior_loss
            positive = eligible & (loss >= positive_start - 2000) & (loss <= positive_end - 2000)

            output = np.full((baseline.height, baseline.width), output_nodata, dtype="uint8")
            output[eligible] = 0
            output[positive] = 1
            profile = baseline.profile.copy()
            for key in ("blockxsize", "blockysize", "tiled", "interleave"):
                profile.pop(key, None)
            profile.update(
                driver="GTiff", count=1, dtype="uint8", nodata=output_nodata,
                compress=labels.get("compression", "DEFLATE"), tiled=False,
            )
            output_path = label_root / aoi / "loss_binary.tif"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
            if temporary.exists():
                temporary.unlink()
            try:
                with rasterio.open(temporary, "w", **profile) as destination:
                    destination.write(output, 1)
                with rasterio.open(temporary) as check:
                    check.read(1)
                temporary.replace(output_path)
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise

            eligible_count = int(eligible.sum())
            positive_count = int(positive.sum())
            manifest_rows.append({
                "aoi": aoi,
                "baseline_year": baseline_year,
                "comparison_year": comparison_year,
                "baseline_composite": str(baseline_path),
                "comparison_composite": str(comparison_path),
                "output_path": str(output_path),
                "crs": str(baseline.crs),
                "width": baseline.width,
                "height": baseline.height,
                "eligible_pixels": eligible_count,
                "positive_pixels": positive_count,
                "negative_pixels": eligible_count - positive_count,
                "positive_pct": round(100 * positive_count / eligible_count, 6) if eligible_count else 0.0,
                "nodata_pixels": int((output == output_nodata).sum()),
            })
            print(f"{aoi}: {positive_count:,} positive / {eligible_count:,} eligible pixels")
            print(f"  Created: {output_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nLabel manifest: {manifest_path}")
    print("Task 3.6 label creation: COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())