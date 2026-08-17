"""Validate Task 3.10 artifacts, leakage controls, and reported metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/baseline_model.yaml")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = resolve(root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_path = resolve(root, config["outputs"]["validation_summary"])
    checks, errors = [], []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    paths = {name: resolve(root, value) for name, value in config["outputs"].items() if name != "validation_summary"}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    check("required_artifacts_exist", not missing, f"missing={missing}")
    if missing:
        result = {"task": "3.10", "status": "FAILED", "checks": checks, "errors": errors}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Validation summary: {output_path}")
        print("Task 3.10 validation: FAILED")
        return 1

    artifact = joblib.load(paths["model_artifact"])
    report = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    predictions = pd.read_csv(paths["predictions"])
    importance = pd.read_csv(paths["feature_importance"])
    manifest = pd.read_csv(resolve(root, config["inputs"]["modeling_manifest"]))
    columns = config["columns"]

    required_keys = {"model", "threshold", "feature_names", "channel_names", "statistics", "target_column"}
    check("model_bundle_schema", required_keys <= set(artifact), f"missing={sorted(required_keys - set(artifact))}")
    threshold = float(artifact["threshold"])
    check("threshold_range", 0.0 <= threshold <= 1.0, f"threshold={threshold}")
    forbidden = set(config["features"].get("forbidden_predictor_columns", []))
    leaked = [name for name in artifact["feature_names"] if name.split("__", 1)[0] in forbidden]
    check("no_target_leakage", not leaked, f"leaked_features={leaked}")
    check("feature_schema_matches_config", artifact["channel_names"] == config["features"]["channel_names"],
          "artifact channel order must match configuration")
    importance_schema_ok = (
        list(importance.columns) == ["feature", "importance"]
        and set(importance["feature"]) == set(artifact["feature_names"])
        and len(importance) == len(artifact["feature_names"])
        and importance["importance"].is_monotonic_decreasing
    )
    check("importance_schema", importance_schema_ok,
          "feature importance file must contain each configured feature once in descending importance order")
    check("importance_sum", np.isclose(importance["importance"].sum(), 1.0, atol=1e-6),
          f"sum={importance['importance'].sum()}")

    prediction_required = {columns["patch_id"], columns["aoi_id"], columns["split"],
                           columns["target"], "probability", "prediction"}
    check("prediction_schema", prediction_required <= set(predictions.columns),
          f"missing={sorted(prediction_required - set(predictions.columns))}")
    check("one_prediction_per_manifest_row", len(predictions) == len(manifest) and
          predictions[columns["patch_id"]].is_unique and
          set(predictions[columns["patch_id"]]) == set(manifest[columns["patch_id"]]),
          f"predictions={len(predictions)}, manifest={len(manifest)}")
    check("probability_range", predictions["probability"].between(0.0, 1.0).all(), "probabilities must be in [0,1]")
    expected_prediction = predictions["probability"].ge(threshold).astype(np.int8)
    check("threshold_applied", predictions["prediction"].astype(np.int8).eq(expected_prediction).all(),
          "predictions must use the saved threshold")

    metric_differences = {}
    for split_name in ("train", "validation", "test"):
        part = predictions[predictions[columns["split"]].eq(split_name)]
        y = part[columns["target"]].astype(np.int8).to_numpy()
        probability = part["probability"].to_numpy()
        predicted = part["prediction"].astype(np.int8).to_numpy()
        recalculated = {
            "roc_auc": roc_auc_score(y, probability), "pr_auc": average_precision_score(y, probability),
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0), "f1": f1_score(y, predicted, zero_division=0),
        }
        reported = report["metrics_by_split"][split_name]
        for name, value in recalculated.items():
            difference = abs(float(value) - float(reported[name]))
            metric_differences[f"{split_name}.{name}"] = difference
    check("reported_metrics_reproduce", max(metric_differences.values(), default=0.0) <= 1e-12,
          f"maximum_absolute_difference={max(metric_differences.values(), default=0.0):.3g}")

    gate_consistent = all(
        bool(item["passed"]) == (float(item["value"]) >= float(item["minimum"]))
        for item in report["quality_gates"].values()
    )
    check("quality_gate_reporting", gate_consistent, f"reported_status={report['quality_gate_status']}")
    result = {
        "task": "3.10", "status": "PASSED" if not errors else "FAILED",
        "quality_gate_status": report["quality_gate_status"], "checks": checks, "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Artifact validation: {result['status']}")
    print(f"Model quality gates: {result['quality_gate_status']}")
    print(f"Validation summary: {output_path}")
    print(f"Task 3.10 validation: {result['status']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
