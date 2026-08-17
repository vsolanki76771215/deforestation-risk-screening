"""SageMaker inference handler for Task 3.14 patch-level Batch Transform."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np


SUPPORTED_STATISTICS = {"mean", "std", "min", "max", "median"}
REQUIRED_ARTIFACT_KEYS = {
    "model", "threshold", "channel_names", "statistics", "archive_key", "expected_patch_size"
}
REQUIRED_INPUT_KEYS = {"patch_id", "aoi_id", "row_off", "col_off", "features"}


def _summarize_patch(array: np.ndarray, statistics: list[str]) -> np.ndarray:
    reducers = {
        "mean": lambda values: np.mean(values, axis=(1, 2)),
        "std": lambda values: np.std(values, axis=(1, 2)),
        "min": lambda values: np.min(values, axis=(1, 2)),
        "max": lambda values: np.max(values, axis=(1, 2)),
        "median": lambda values: np.median(values, axis=(1, 2)),
    }
    unsupported = set(statistics) - SUPPORTED_STATISTICS
    if unsupported:
        raise ValueError(f"Unsupported model statistics: {sorted(unsupported)}")
    return np.stack([reducers[name](array) for name in statistics], axis=1).reshape(-1).astype(np.float32)


def model_fn(model_dir: str):
    artifact_path = Path(model_dir) / "model.joblib"
    artifact = joblib.load(artifact_path)
    missing = REQUIRED_ARTIFACT_KEYS - set(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing required keys: {sorted(missing)}")
    threshold = float(artifact["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Model threshold must be in [0, 1]")
    artifact["threshold"] = threshold
    artifact["expected_shape"] = (
        len(artifact["channel_names"]), int(artifact["expected_patch_size"]), int(artifact["expected_patch_size"])
    )
    return artifact


def input_fn(request_body, request_content_type: str):
    if request_content_type not in {"application/json", "application/jsonlines"}:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body)
    missing = REQUIRED_INPUT_KEYS - set(payload)
    if missing:
        raise ValueError(f"Input record is missing keys: {sorted(missing)}")
    return payload


def predict_fn(input_data: dict, artifact: dict) -> dict:
    array = np.asarray(input_data["features"], dtype=np.float32)
    if array.shape != artifact["expected_shape"]:
        raise ValueError(f"Expected feature patch shape {artifact['expected_shape']}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Feature patch contains non-finite values")
    vector = _summarize_patch(array, list(artifact["statistics"]))
    model = artifact["model"]
    expected = getattr(model, "n_features_in_", vector.shape[0])
    if vector.shape[0] != expected:
        raise ValueError(f"Model expects {expected} features but patch yields {vector.shape[0]}")
    probability = float(model.predict_proba(vector.reshape(1, -1))[0, 1])
    threshold = artifact["threshold"]
    return {
        "patch_id": str(input_data["patch_id"]), "aoi_id": str(input_data["aoi_id"]),
        "row_off": int(input_data["row_off"]), "col_off": int(input_data["col_off"]),
        "probability": probability, "prediction": int(probability >= threshold),
        "threshold": threshold, "quality_flag": "accepted",
    }


def output_fn(prediction: dict, accept: str):
    if accept not in {"application/json", "application/jsonlines", "*/*"}:
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction, separators=(",", ":")), "application/json"
