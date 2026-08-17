#!/usr/bin/env python3
"""Acquire selected Sentinel-2 AOI windows and build SCL-masked composites."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds, transform_geom

BANDS = ("B02", "B03", "B04", "B08")
KEY = ("aoi_id", "target_year", "item_id")
ACQUIRED_FIELDS = [
    "aoi_id", "aoi_role", "target_year", "item_id", "selection_order",
    "asset", "source_href", "local_path", "sha256", "width", "height",
    "crs", "resolution_m", "nodata", "status",
]
COMPOSITE_FIELDS = [
    "aoi_id", "aoi_role", "target_year", "scene_count", "item_ids",
    "composite_path", "valid_mask_path", "width", "height", "crs",
    "resolution_m", "valid_pixel_count", "aoi_pixel_count",
    "valid_coverage_pct", "sha256", "status",
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


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[name] for name in KEY)  # type: ignore[return-value]


def expected_aoi_years(config: dict[str, Any]) -> set[tuple[str, int]]:
    """Return every AOI/year combination required by the configuration."""
    s2 = config["sentinel2"]
    years = {int(s2["baseline"]["year"]), int(s2["comparison"]["year"])}
    if len(years) != 2:
        raise ValueError("Sentinel-2 baseline and comparison years must be different")
    return {(aoi_id, year) for aoi_id in config["aois"] for year in years}


def resolve_selected(selected: list[dict[str, str]], inventory: list[dict[str, str]]) -> list[dict[str, str]]:
    index: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in inventory:
        index[row_key(row)].append(row)
    resolved = []
    seen = set()
    for selection in selected:
        key = row_key(selection)
        if key in seen:
            raise ValueError(f"Duplicate selected input: {key}")
        seen.add(key)
        matches = index.get(key, [])
        if len(matches) != 1:
            raise ValueError(f"Selected input must resolve exactly once: {key}; matches={len(matches)}")
        source = matches[0]
        missing = [asset for asset in (*BANDS, "SCL") if not source.get(f"{asset}_href")]
        if missing:
            raise ValueError(f"Missing assets for {key}: {missing}")
        resolved.append({**source, **selection, **{f"{a}_href": source[f"{a}_href"] for a in (*BANDS, "SCL")}})
    return resolved


def sas_token(catalog_url: str, collection: str) -> str:
    parsed = urllib.parse.urlsplit(catalog_url)
    url = f"{parsed.scheme}://{parsed.netloc}/api/sas/v1/token/{collection}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["token"]


def signed_href(href: str, token: str) -> str:
    return f"{href}{'&' if '?' in href else '?'}{token}"


def grid(bounds: list[float], crs: str, resolution: float):
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bounds)
    left, bottom = math.floor(left / resolution) * resolution, math.floor(bottom / resolution) * resolution
    right, top = math.ceil(right / resolution) * resolution, math.ceil(top / resolution) * resolution
    width, height = int(round((right - left) / resolution)), int(round((top - bottom) / resolution))
    transform = from_origin(left, top, resolution, resolution)
    polygon = {"type": "Polygon", "coordinates": [[
        [bounds[0], bounds[1]], [bounds[2], bounds[1]], [bounds[2], bounds[3]],
        [bounds[0], bounds[3]], [bounds[0], bounds[1]],
    ]]}
    projected = transform_geom("EPSG:4326", crs, polygon)
    mask = geometry_mask([projected], out_shape=(height, width), transform=transform, invert=True)
    return transform, width, height, mask


def read_to_grid(href: str, token: str, transform, crs: str, width: int, height: int,
                 dtype: str, nodata: float, resampling: Resampling) -> np.ndarray:
    destination = np.full((height, width), nodata, dtype=dtype)
    for attempt in range(3):
        try:
            with rasterio.Env(GDAL_HTTP_UNSAFESSL="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open(signed_href(href, token)) as src:
                    reproject(rasterio.band(src, 1), destination, src_transform=src.transform,
                              src_crs=src.crs, src_nodata=src.nodata,
                              dst_transform=transform, dst_crs=crs, dst_nodata=nodata,
                              resampling=resampling)
            return destination
        except rasterio.errors.RasterioError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Raster read failed")


def write_tif(path: Path, array: np.ndarray, transform, crs: str, nodata: float,
              descriptions: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = array if array.ndim == 3 else array[np.newaxis, ...]
    profile = {"driver": "GTiff", "height": data.shape[1], "width": data.shape[2],
               "count": data.shape[0], "dtype": data.dtype, "crs": crs,
               "transform": transform, "nodata": nodata, "compress": "deflate",
               "tiled": True, "blockxsize": 256, "blockysize": 256}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            dst.descriptions = descriptions


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def build(config: dict[str, Any], rows: list[dict[str, str]], token: str):
    s2, outputs = config["sentinel2"], config["outputs"]
    policy = s2["compositing"]
    crs = config["specification"]["analysis_crs"]
    resolution, scale, output_nodata = float(policy["target_grid_resolution_m"]), float(policy["reflectance_scale_factor"]), float(policy["output_nodata"])
    excluded = set(map(int, s2["excluded_scl_classes"]))
    acquired, composites = [], []
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["aoi_id"], int(row["target_year"]))].append(row)
    for (aoi_id, year), scenes in sorted(groups.items()):
        aoi = config["aois"][aoi_id]
        transform, width, height, aoi_mask = grid(aoi["bounds_epsg4326"], crs, resolution)
        scene_stacks = []
        for scene in sorted(scenes, key=lambda r: int(r.get("selection_order") or 0)):
            scene_dir = Path(outputs["sentinel2_raw_root"]) / aoi_id / str(year) / scene["item_id"]
            scl = read_to_grid(scene["SCL_href"], token, transform, crs, width, height,
                               "uint8", 255, Resampling.nearest)
            valid = aoi_mask & (scl != 255) & ~np.isin(scl, list(excluded))
            scl_path = scene_dir / "SCL_10m.tif"
            write_tif(scl_path, scl, transform, crs, 255)
            acquired.append(asset_row(scene, "SCL", scene["SCL_href"], scl_path, width, height, crs, resolution, 255))
            arrays = []
            for band in BANDS:
                raw = read_to_grid(scene[f"{band}_href"], token, transform, crs, width, height,
                                   "uint16", 0, Resampling.bilinear)
                band_path = scene_dir / f"{band}_10m.tif"
                write_tif(band_path, raw, transform, crs, 0)
                acquired.append(asset_row(scene, band, scene[f"{band}_href"], band_path, width, height, crs, resolution, 0))
                arrays.append(np.where(valid & (raw > 0), raw.astype("float32") * scale, np.nan))
            scene_stacks.append(np.stack(arrays))
        stack = np.stack(scene_stacks)
        with np.errstate(all="ignore"):
            composite = np.nanmedian(stack, axis=0).astype("float32")
        valid_composite = np.all(np.isfinite(composite), axis=0) & aoi_mask
        composite[:, ~valid_composite] = output_nodata
        out_dir = Path(outputs["sentinel2_composite_root"]) / aoi_id
        composite_path, mask_path = out_dir / f"sentinel2_{year}_composite.tif", out_dir / f"sentinel2_{year}_valid_mask.tif"
        write_tif(composite_path, composite, transform, crs, output_nodata, BANDS)
        write_tif(mask_path, valid_composite.astype("uint8"), transform, crs, 0)
        valid_count, aoi_count = int(valid_composite.sum()), int(aoi_mask.sum())
        coverage = 100.0 * valid_count / aoi_count
        composites.append({"aoi_id": aoi_id, "aoi_role": aoi["role"], "target_year": year,
            "scene_count": len(scenes), "item_ids": ";".join(r["item_id"] for r in scenes),
            "composite_path": str(composite_path), "valid_mask_path": str(mask_path),
            "width": width, "height": height, "crs": crs, "resolution_m": resolution,
            "valid_pixel_count": valid_count, "aoi_pixel_count": aoi_count,
            "valid_coverage_pct": round(coverage, 4), "sha256": digest(composite_path),
            "status": "passed" if coverage >= float(s2["minimum_valid_coverage_pct"]) else "failed"})
    return acquired, composites


def asset_row(scene, asset, href, path, width, height, crs, resolution, nodata):
    return {"aoi_id": scene["aoi_id"], "aoi_role": scene["aoi_role"],
            "target_year": scene["target_year"], "item_id": scene["item_id"],
            "selection_order": scene.get("selection_order", ""), "asset": asset,
            "source_href": href, "local_path": str(path), "sha256": digest(path),
            "width": width, "height": height, "crs": crs, "resolution_m": resolution,
            "nodata": nodata, "status": "acquired"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/geospatial.yaml"))
    parser.add_argument("--selected-manifest", type=Path)
    parser.add_argument("--inventory-manifest", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    selected_path = args.selected_manifest or Path(outputs["sentinel2_composite_inputs"])
    inventory_path = args.inventory_manifest or Path(outputs["sentinel2_manifest"])
    resolved = resolve_selected(read_rows(selected_path), read_rows(inventory_path))
    expected_groups = expected_aoi_years(config)
    actual_groups = {(row["aoi_id"], int(row["target_year"])) for row in resolved}
    missing_groups = sorted(expected_groups - actual_groups)
    unexpected_groups = sorted(actual_groups - expected_groups)
    if missing_groups or unexpected_groups:
        raise ValueError(
            "Selected AOI/year inputs do not match the configuration; "
            f"missing={missing_groups}, unexpected={unexpected_groups}"
        )
    token = sas_token(config["sentinel2"]["catalog_url"], config["sentinel2"]["collection"])
    acquired, composites = build(config, resolved, token)
    write_rows(Path(outputs["sentinel2_acquisition_manifest"]), ACQUIRED_FIELDS, acquired)
    write_rows(Path(outputs["sentinel2_composite_manifest"]), COMPOSITE_FIELDS, composites)
    assets_per_scene = len(BANDS) + 1  # Reflectance bands plus SCL.
    expected_asset_count = len(resolved) * assets_per_scene
    composite_groups = {(row["aoi_id"], int(row["target_year"])) for row in composites}
    passed = (
        len(acquired) == expected_asset_count
        and len(composites) == len(expected_groups)
        and composite_groups == expected_groups
        and all(row["status"] == "passed" for row in composites)
    )
    summary = {"task": "3.4-acquire-and-build-sentinel2-composites", "completed_at_utc": utc_now(),
               "status": "passed" if passed else "failed", "selected_scene_count": len(resolved),
               "expected_aoi_year_count": len(expected_groups),
               "expected_acquired_asset_count": expected_asset_count,
               "acquired_asset_count": len(acquired), "composite_count": len(composites),
               "bands": list(BANDS), "excluded_scl_classes": config["sentinel2"]["excluded_scl_classes"],
               "results": composites}
    Path(outputs["sentinel2_composite_summary"]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in composites:
        print(f"{row['aoi_id']:24} {row['target_year']}: {row['valid_coverage_pct']:.2f}% [{row['status']}]")
    print(f"Task 3.4 validation: {summary['status'].upper()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
