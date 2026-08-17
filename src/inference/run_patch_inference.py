#!/usr/bin/env python3
"""Run the approved patch model and write prediction, map, and report outputs.

The input manifest must contain ``patch_id``, ``aoi_id``, and
``feature_patch_path``.  For map output it must also contain ``row_off`` and
``col_off``; the reference feature raster supplies the CRS and affine transform
used to turn each patch footprint into a GeoJSON polygon.

This script deliberately consumes only the model artifact and new feature
patches.  It never reads labels or label-derived fields, so it is safe to use
for an unlabeled future Sentinel-2 inference run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


REQUIRED_COLUMNS = ("patch_id", "aoi_id", "feature_patch_path", "row_off", "col_off")
SUPPORTED_STATISTICS = {"mean", "std", "min", "max", "median"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-artifact", required=True, help="joblib model artifact produced by Task 3.14")
    parser.add_argument("--input-manifest", required=True, help="CSV of new Sentinel-2 feature patch files")
    parser.add_argument("--reference-raster", required=True, help="Aligned 11-band feature-stack GeoTIFF")
    parser.add_argument("--output-dir", required=True, help="Directory for this immutable inference run")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", default=None, help="Optional stable run identifier")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_patch(array: np.ndarray, statistics: list[str]) -> np.ndarray:
    reducers = {
        "mean": lambda values: np.mean(values, axis=(1, 2)),
        "std": lambda values: np.std(values, axis=(1, 2)),
        "min": lambda values: np.min(values, axis=(1, 2)),
        "max": lambda values: np.max(values, axis=(1, 2)),
        "median": lambda values: np.median(values, axis=(1, 2)),
    }
    unknown = set(statistics) - SUPPORTED_STATISTICS
    if unknown:
        raise ValueError(f"Unsupported model statistics: {sorted(unknown)}")
    return np.stack([reducers[name](array) for name in statistics], axis=1).reshape(-1).astype(np.float32)


def load_artifact(path: Path) -> dict:
    artifact = joblib.load(path)
    required = {"model", "threshold", "channel_names", "statistics", "archive_key", "expected_patch_size"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing required keys: {sorted(missing)}")
    if not 0.0 <= float(artifact["threshold"]) <= 1.0:
        raise ValueError("Model threshold must be in [0, 1]")
    if set(artifact["statistics"]) - SUPPORTED_STATISTICS:
        raise ValueError("Model uses unsupported feature statistics")
    return artifact


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Input manifest is missing columns: {sorted(missing)}")
    if frame.empty or frame.patch_id.astype(str).duplicated().any():
        raise ValueError("Input manifest must be non-empty and have unique patch_id values")
    for column in ("row_off", "col_off"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if (frame[column] < 0).any():
            raise ValueError(f"{column} cannot be negative")
    return frame


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    os.replace(temp, path)


def patch_polygon(transform, row_off: int, col_off: int, size: int, transformer: Transformer) -> list[list[float]]:
    corners = [(col_off, row_off), (col_off + size, row_off), (col_off + size, row_off + size), (col_off, row_off + size)]
    coordinates = []
    for col, row in corners:
        x, y = transform * (col, row)
        lon, lat = transformer.transform(x, y)
        coordinates.append([round(lon, 8), round(lat, 8)])
    return coordinates + [coordinates[0]]


def write_geojson(path: Path, predictions: pd.DataFrame, transform, crs, patch_size: int) -> None:
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    features = []
    for row in predictions.itertuples(index=False):
        properties = {
            "patch_id": str(row.patch_id), "aoi_id": str(row.aoi_id),
            "probability": round(float(row.probability), 7), "prediction": int(row.prediction),
            "threshold": round(float(row.threshold), 7), "quality_flag": str(row.quality_flag),
        }
        features.append({"type": "Feature", "properties": properties, "geometry": {
            "type": "Polygon", "coordinates": [patch_polygon(transform, int(row.row_off), int(row.col_off), patch_size, transformer)],
        }})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    artifact_path = resolve(root, args.model_artifact)
    manifest_path = resolve(root, args.input_manifest)
    raster_path = resolve(root, args.reference_raster)
    output_dir = resolve(root, args.output_dir)
    for path in (artifact_path, manifest_path, raster_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(artifact_path)
    frame = read_manifest(manifest_path)
    channel_count = len(artifact["channel_names"])
    patch_size = int(artifact["expected_patch_size"])
    statistics = list(artifact["statistics"])
    archive_key = str(artifact["archive_key"])
    expected_shape = (channel_count, patch_size, patch_size)

    accepted, rejected, vectors = [], [], []
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        record = row._asdict()
        patch_path = resolve(root, str(record["feature_patch_path"]))
        try:
            with np.load(patch_path, allow_pickle=False) as archive:
                if archive_key not in archive:
                    raise KeyError(f"missing {archive_key!r} array")
                array = archive[archive_key]
            if array.shape != expected_shape:
                raise ValueError(f"expected shape {expected_shape}, found {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError("contains non-finite values")
            vectors.append(summarize_patch(array, statistics))
            accepted.append(record)
        except (OSError, KeyError, ValueError) as error:
            rejected.append({**record, "quality_flag": "rejected", "rejection_reason": str(error)})
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(frame)):
            print(f"Validated {index}/{len(frame)} patches", flush=True)
    if not accepted:
        raise RuntimeError("No valid feature patches remained after validation")

    matrix = np.vstack(vectors)
    model = artifact["model"]
    expected_features = getattr(model, "n_features_in_", matrix.shape[1])
    if matrix.shape[1] != expected_features:
        raise ValueError(f"Model expects {expected_features} features but patches yield {matrix.shape[1]}")
    probability = model.predict_proba(matrix)[:, 1]
    threshold = float(artifact["threshold"])
    predictions = pd.DataFrame(accepted)
    predictions["probability"] = probability
    predictions["prediction"] = (probability >= threshold).astype(np.int8)
    predictions["threshold"] = threshold
    predictions["quality_flag"] = "accepted"

    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            raise ValueError("Reference raster has no CRS")
        if raster.count != channel_count:
            raise ValueError(f"Reference raster has {raster.count} bands; model requires {channel_count}")
        if ((predictions.row_off + patch_size > raster.height) | (predictions.col_off + patch_size > raster.width)).any():
            raise ValueError("A manifest patch footprint falls outside the reference raster")
        write_geojson(output_dir / "prediction_map.geojson", predictions, raster.transform, raster.crs, patch_size)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("inference-%Y%m%dT%H%M%SZ")
    predictions.insert(0, "run_id", run_id)
    atomic_csv(output_dir / "predictions.csv", predictions)
    rejected_frame = pd.DataFrame(rejected)
    if rejected_frame.empty:
        rejected_frame = pd.DataFrame(columns=[*frame.columns, "quality_flag", "rejection_reason"])
    atomic_csv(output_dir / "rejected_patches.csv", rejected_frame)
    aoi_report = predictions.groupby("aoi_id", dropna=False).agg(
        patch_count=("patch_id", "size"), alert_count=("prediction", "sum"),
        mean_probability=("probability", "mean"), max_probability=("probability", "max"),
    ).reset_index()
    aoi_report["alert_rate"] = aoi_report.alert_count / aoi_report.patch_count
    atomic_csv(output_dir / "aoi_summary.csv", aoi_report)
    report = {
        "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED", "task_scope": "deforestation-risk probability; not confirmed illegal-mining detection",
        "input_manifest": str(manifest_path), "input_manifest_sha256": sha256(manifest_path),
        "reference_raster": str(raster_path), "model_artifact": str(artifact_path),
        "model_artifact_sha256": sha256(artifact_path), "threshold": threshold,
        "expected_patch_shape": expected_shape, "accepted_patch_count": len(predictions),
        "rejected_patch_count": len(rejected_frame), "alert_count": int(predictions.prediction.sum()),
        "outputs": {"predictions": "predictions.csv", "rejected": "rejected_patches.csv", "aoi_summary": "aoi_summary.csv", "map": "prediction_map.geojson"},
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Predicted {len(predictions)} valid patches; alerts={int(predictions.prediction.sum())}; rejected={len(rejected_frame)}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
