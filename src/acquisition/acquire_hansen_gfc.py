from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
import yaml
from rasterio.features import geometry_mask
from rasterio.merge import merge


def download_file(url: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already downloaded: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".part")

    print(f"Downloading: {url}")

    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()

        total_bytes = int(response.headers.get("content-length", 0))
        downloaded_bytes = 0

        with temporary_path.open("wb") as destination:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                destination.write(chunk)
                downloaded_bytes += len(chunk)

                if total_bytes:
                    percent = downloaded_bytes / total_bytes * 100
                    print(
                        f"\r  {downloaded_bytes / 1024**2:.1f} MB "
                        f"of {total_bytes / 1024**2:.1f} MB "
                        f"({percent:.1f}%)",
                        end="",
                    )

    print()
    temporary_path.replace(output_path)


def source_filename(version: str, layer: str, tile: str) -> str:
    return f"Hansen_{version}_{layer}_{tile}.tif"


def clip_layer(
    source_paths: list[Path],
    aoi_path: Path,
    output_path: Path,
    nodata_value: int,
    compression: str,
) -> dict:
    aoi = gpd.read_file(aoi_path)

    if aoi.empty:
        raise ValueError(f"No geometry found in {aoi_path}")

    if aoi.crs is None:
        raise ValueError(f"AOI has no CRS: {aoi_path}")

    datasets = [rasterio.open(path) for path in source_paths]

    try:
        raster_crs = datasets[0].crs

        if raster_crs is None:
            raise ValueError(f"Source raster has no CRS: {source_paths[0]}")

        aoi = aoi.to_crs(raster_crs)
        bounds = tuple(aoi.total_bounds)

        mosaic, transform = merge(
            datasets,
            bounds=bounds,
            nodata=nodata_value,
            dtype="uint8",
        )

        inside_aoi = geometry_mask(
            list(aoi.geometry),
            out_shape=(mosaic.shape[1], mosaic.shape[2]),
            transform=transform,
            invert=True,
        )

        clipped = mosaic[0].astype("uint8")
        clipped[~inside_aoi] = nodata_value

        profile = datasets[0].profile.copy()

        # Remove source tiling settings that may be incompatible with
        # the dimensions of the clipped raster.
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)
        profile.pop("interleave", None)

        profile.update(
            driver="GTiff",
            count=1,
            width=clipped.shape[1],
            height=clipped.shape[0],
            transform=transform,
            crs=raster_crs,
            dtype="uint8",
            nodata=nodata_value,
            compress=compression,
            tiled=False,
            BIGTIFF="IF_SAFER",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(clipped, 1)

        valid = clipped != nodata_value
        valid_values = clipped[valid]

        return {
            "crs": str(raster_crs),
            "width": clipped.shape[1],
            "height": clipped.shape[0],
            "dtype": "uint8",
            "nodata": nodata_value,
            "valid_pixels": int(valid.sum()),
            "min_value": (
                int(valid_values.min()) if valid_values.size else None
            ),
            "max_value": (
                int(valid_values.max()) if valid_values.size else None
            ),
        }
    finally:
        for dataset in datasets:
            dataset.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config/geospatial.yaml",
    )
    parser.add_argument(
        "--aoi-dir",
        required=True,
        help="Directory containing AOI GeoJSON files",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    aoi_directory = Path(args.aoi_dir)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    hansen = config["hansen"]
    outputs = config["outputs"]

    version = hansen["version"]
    base_url = hansen["base_url"].rstrip("/")
    tiles = hansen["tiles"]

    # Only these layers produce the six requested files.
    layers = ["treecover2000", "lossyear"]

    raw_root = Path(outputs["hansen_raw_root"]) / version
    clipped_root = Path(outputs["hansen_clipped_root"])
    manifest_path = Path(outputs["hansen_clipped_manifest"])

    nodata_value = int(hansen.get("output_nodata", 255))
    compression = hansen.get("compression", "DEFLATE")

    source_files: dict[str, list[Path]] = {}

    for layer in layers:
        source_files[layer] = []

        for tile in tiles:
            filename = source_filename(version, layer, tile)
            source_path = raw_root / filename
            url = f"{base_url}/{filename}"

            download_file(url, source_path)
            source_files[layer].append(source_path)

    manifest_rows = []

    for aoi_name in config["aois"]:
        aoi_path = aoi_directory / f"{aoi_name}.geojson"

        if not aoi_path.exists():
            raise FileNotFoundError(f"Missing AOI file: {aoi_path}")

        for layer in layers:
            output_path = clipped_root / aoi_name / f"{layer}.tif"

            print(f"Clipping {layer} to {aoi_name}...")

            statistics = clip_layer(
                source_paths=source_files[layer],
                aoi_path=aoi_path,
                output_path=output_path,
                nodata_value=nodata_value,
                compression=compression,
            )

            manifest_rows.append(
                {
                    "aoi": aoi_name,
                    "layer": layer,
                    "version": version,
                    "source_tiles": ";".join(tiles),
                    "output_path": str(output_path),
                    **statistics,
                }
            )

            print(f"Created: {output_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "aoi",
        "layer",
        "version",
        "source_tiles",
        "output_path",
        "crs",
        "width",
        "height",
        "dtype",
        "nodata",
        "valid_pixels",
        "min_value",
        "max_value",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest created: {manifest_path}")
    print("Hansen acquisition and clipping completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())