"""Train and evaluate the Task 3.10 CPU Random Forest baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/baseline_model.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def feature_names(channels: list[str], statistics: list[str]) -> list[str]:
    return [f"{channel}__{stat}" for channel in channels for stat in statistics]


def summarize_patch(array: np.ndarray, statistics: list[str]) -> np.ndarray:
    reducers = {
        "mean": lambda values: np.mean(values, axis=(1, 2)),
        "std": lambda values: np.std(values, axis=(1, 2)),
        "min": lambda values: np.min(values, axis=(1, 2)),
        "max": lambda values: np.max(values, axis=(1, 2)),
        "median": lambda values: np.median(values, axis=(1, 2)),
    }
    unknown = [name for name in statistics if name not in reducers]
    if unknown:
        raise ValueError(f"Unsupported feature statistics: {unknown}")
    # Channel-major order must match feature_names().
    reduced = np.stack([reducers[name](array) for name in statistics], axis=1)
    return reduced.reshape(-1).astype(np.float32, copy=False)


def load_features(
    frame: pd.DataFrame,
    root: Path,
    path_column: str,
    archive_key: str,
    channel_count: int,
    patch_size: int,
    statistics: list[str],
    progress_every: int,
) -> np.ndarray:
    matrix = np.empty((len(frame), channel_count * len(statistics)), dtype=np.float32)
    total = len(frame)
    for index, value in enumerate(frame[path_column].astype(str), start=1):
        path = resolve(root, value)
        with np.load(path, allow_pickle=False) as archive:
            if archive_key not in archive:
                raise KeyError(f"{path} has no {archive_key!r} array")
            array = archive[archive_key]
        expected = (channel_count, patch_size, patch_size)
        if array.shape != expected:
            raise ValueError(f"{path}: expected feature shape {expected}, found {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: feature array contains non-finite values")
        matrix[index - 1] = summarize_patch(array, statistics)
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(f"Feature extraction: {index}/{total} patches ({100.0 * index / total:.1f}%)", flush=True)
    return matrix


def select_threshold(y_true: np.ndarray, probability: np.ndarray, minimum_recall: float, points: int) -> dict:
    if points < 2:
        raise ValueError("threshold_selection.grid_points must be at least 2")
    candidates = []
    for threshold in np.linspace(0.0, 1.0, points):
        predicted = probability >= threshold
        recall = recall_score(y_true, predicted, zero_division=0)
        precision = precision_score(y_true, predicted, zero_division=0)
        f1 = f1_score(y_true, predicted, zero_division=0)
        candidates.append((threshold, recall, precision, f1))
    eligible = [row for row in candidates if row[1] >= minimum_recall]
    pool = eligible or candidates
    # Highest F1, then precision, then threshold gives deterministic selection.
    threshold, recall, precision, f1 = max(pool, key=lambda row: (row[3], row[2], row[0]))
    return {
        "threshold": float(threshold),
        "validation_recall": float(recall),
        "validation_precision": float(precision),
        "validation_f1": float(f1),
        "minimum_recall_constraint_met": bool(eligible),
    }


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "row_count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "negative_count": int(len(y_true) - y_true.sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    config_path = resolve(root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = resolve(root, config["inputs"]["modeling_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Modeling manifest not found: {manifest_path}")

    outputs = {name: resolve(root, value) for name, value in config["outputs"].items() if name != "validation_summary"}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Task 3.10 output exists; pass --overwrite to replace it: {existing}")

    frame = pd.read_csv(manifest_path)
    columns = config["columns"]
    required = [columns[name] for name in ("patch_id", "aoi_id", "feature_patch_path", "target", "split")]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Modeling manifest is missing columns: {missing}")
    if frame.empty or frame[columns["patch_id"]].duplicated().any():
        raise ValueError("Modeling manifest must be non-empty with unique patch IDs")

    split_column = columns["split"]
    actual_splits = set(frame[split_column].astype(str))
    if actual_splits != {"train", "validation", "test"}:
        raise ValueError(f"Expected train/validation/test splits, found {sorted(actual_splits)}")
    target_column = columns["target"]
    target = pd.to_numeric(frame[target_column], errors="raise").astype(np.int8)
    if not target.isin([0, 1]).all():
        raise ValueError(f"{target_column} must be binary")
    for split_name in ("train", "validation", "test"):
        if target[frame[split_column].eq(split_name)].nunique() != 2:
            raise ValueError(f"Split {split_name} must contain both target classes")

    feature_cfg = config["features"]
    forbidden = set(feature_cfg.get("forbidden_predictor_columns", []))
    channels = list(feature_cfg["channel_names"])
    if forbidden.intersection(channels):
        raise ValueError(f"Target-derived predictors are forbidden: {sorted(forbidden.intersection(channels))}")
    statistics = list(feature_cfg["statistics"])
    names = feature_names(channels, statistics)
    matrix = load_features(
        frame, root, columns["feature_patch_path"], str(feature_cfg["archive_key"]),
        len(channels), int(feature_cfg["expected_patch_size"]), statistics, args.progress_every,
    )

    masks = {name: frame[split_column].eq(name).to_numpy() for name in ("train", "validation", "test")}
    y = target.to_numpy()
    model_cfg = config["model"]
    if model_cfg["type"] != "RandomForestClassifier":
        raise ValueError("Task 3.10 baseline supports RandomForestClassifier")
    model = RandomForestClassifier(
        n_estimators=int(model_cfg["n_estimators"]), max_depth=model_cfg.get("max_depth"),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]), max_features=model_cfg["max_features"],
        class_weight=model_cfg["class_weight"], n_jobs=int(model_cfg["n_jobs"]),
        random_state=int(config["random_seed"]),
    )
    print(f"Training Random Forest on {int(masks['train'].sum())} patches...", flush=True)
    model.fit(matrix[masks["train"]], y[masks["train"]])

    probabilities = {name: model.predict_proba(matrix[mask])[:, 1] for name, mask in masks.items()}
    threshold_result = select_threshold(
        y[masks["validation"]], probabilities["validation"],
        float(config["threshold_selection"]["minimum_recall"]),
        int(config["threshold_selection"]["grid_points"]),
    )
    threshold = threshold_result["threshold"]
    split_metrics = {name: metrics(y[masks[name]], probabilities[name], threshold) for name in masks}
    gates = config["quality_gates"]
    gate_results = {
        "test_roc_auc": {"value": split_metrics["test"]["roc_auc"], "minimum": float(gates["test_roc_auc_min"])},
        "test_pr_auc": {"value": split_metrics["test"]["pr_auc"], "minimum": float(gates["test_pr_auc_min"])},
        "test_recall": {"value": split_metrics["test"]["recall"], "minimum": float(gates["test_recall_min"])},
    }
    for result in gate_results.values():
        result["passed"] = result["value"] >= result["minimum"]

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model, "threshold": threshold, "feature_names": names,
        "channel_names": channels, "statistics": statistics,
        "archive_key": feature_cfg["archive_key"], "expected_patch_size": int(feature_cfg["expected_patch_size"]),
        "target_column": target_column, "random_seed": int(config["random_seed"]),
        "manifest_sha256": sha256(manifest_path), "config_sha256": sha256(config_path),
    }
    joblib.dump(artifact, outputs["model_artifact"], compress=3)

    prediction_parts = []
    for split_name, mask in masks.items():
        part = frame.loc[mask, [columns["patch_id"], columns["aoi_id"], split_column, target_column]].copy()
        part["probability"] = probabilities[split_name]
        part["prediction"] = (probabilities[split_name] >= threshold).astype(np.int8)
        prediction_parts.append(part)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(outputs["predictions"], index=False, lineterminator="\n")
    pd.DataFrame({"feature": names, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(outputs["feature_importance"], index=False, lineterminator="\n")

    result = {
        "task": "3.10", "status": "COMPLETED", "model_type": model_cfg["type"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": config["inputs"]["modeling_manifest"], "manifest_sha256": artifact["manifest_sha256"],
        "config_sha256": artifact["config_sha256"], "random_seed": int(config["random_seed"]),
        "feature_count": len(names), "features": names, "threshold_selection": threshold_result,
        "metrics_by_split": split_metrics, "quality_gates": gate_results,
        "quality_gate_status": "PASSED" if all(x["passed"] for x in gate_results.values()) else "FAILED",
        "artifacts": {name: config["outputs"][name] for name in outputs},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    outputs["metrics"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Selected threshold: {threshold:.3f}")
    for name in ("validation", "test"):
        item = split_metrics[name]
        print(f"{name}: ROC-AUC={item['roc_auc']:.4f}, PR-AUC={item['pr_auc']:.4f}, "
              f"precision={item['precision']:.4f}, recall={item['recall']:.4f}, F1={item['f1']:.4f}")
    print(f"Quality gates: {result['quality_gate_status']}")
    print(f"Metrics: {outputs['metrics']}")
    print("Task 3.10 baseline training: COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())