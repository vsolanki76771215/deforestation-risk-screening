#!/usr/bin/env python3
"""Validate the files emitted by ``run_patch_inference.py``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.output_dir)
    required = {name: directory / name for name in (
        "predictions.csv", "rejected_patches.csv", "aoi_summary.csv", "prediction_map.geojson", "run_report.json"
    )}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inference output(s): {missing}")
    predictions = pd.read_csv(required["predictions.csv"])
    expected_columns = {"run_id", "patch_id", "aoi_id", "row_off", "col_off", "probability", "prediction", "threshold", "quality_flag"}
    if predictions.empty or expected_columns - set(predictions.columns):
        raise ValueError("Prediction table is empty or missing required columns")
    if predictions.patch_id.duplicated().any() or not predictions.probability.between(0, 1).all():
        raise ValueError("Prediction IDs must be unique and probabilities must be in [0, 1]")
    if not set(predictions.prediction.unique()).issubset({0, 1}) or not (predictions.quality_flag == "accepted").all():
        raise ValueError("Predictions must be binary accepted records")
    if not ((predictions.probability >= predictions.threshold) == predictions.prediction.astype(bool)).all():
        raise ValueError("Predictions do not match the saved threshold")
    geojson = json.loads(required["prediction_map.geojson"].read_text(encoding="utf-8"))
    features = geojson.get("features", [])
    if geojson.get("type") != "FeatureCollection" or len(features) != len(predictions):
        raise ValueError("GeoJSON must contain one feature for each accepted prediction")
    if {str(item["properties"]["patch_id"]) for item in features} != set(predictions.patch_id.astype(str)):
        raise ValueError("GeoJSON patch IDs do not match predictions")
    report = json.loads(required["run_report.json"].read_text(encoding="utf-8"))
    if report.get("accepted_patch_count") != len(predictions) or report.get("alert_count") != int(predictions.prediction.sum()):
        raise ValueError("Run report counts do not match predictions")
    print(f"Inference outputs: PASSED ({len(predictions)} accepted patches; {int(predictions.prediction.sum())} alerts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
