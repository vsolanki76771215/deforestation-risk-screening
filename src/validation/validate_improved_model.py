"""Validate Task 3.11 artifacts, metric reproduction, and leakage safeguards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/improved_model.yaml")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config = yaml.safe_load(resolve(root, args.config).read_text(encoding="utf-8"))
    summary_path = resolve(root, config["outputs"]["validation_summary"])
    checks, errors = [], []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    paths = {k: resolve(root, v) for k, v in config["outputs"].items() if k != "validation_summary"}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    check("required_artifacts_exist", not missing, f"missing={missing}")
    if missing:
        result = {"task": "3.11", "status": "FAILED", "checks": checks, "errors": errors}
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Validation summary: {summary_path}")
        print("Task 3.11 validation: FAILED")
        return 1

    artifact = joblib.load(paths["model_artifact"])
    report = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    predictions = pd.read_csv(paths["predictions"])
    tuning = pd.read_csv(paths["tuning_results"])
    importance = pd.read_csv(paths["feature_importance"])
    manifest = pd.read_csv(resolve(root, config["inputs"]["modeling_manifest"]))
    columns = config["columns"]
    required_keys = {"model", "threshold", "feature_names", "channel_names", "statistics",
                     "selected_candidate_id", "selection_used_splits", "test_used_for_selection"}
    check("model_bundle_schema", required_keys <= set(artifact), f"missing={sorted(required_keys - set(artifact))}")
    check("test_not_used_for_selection", artifact.get("selection_used_splits") == ["train", "validation"] and
          artifact.get("test_used_for_selection") is False and
          report.get("selection_protocol", {}).get("test_used_for_selection") is False,
          "selection must use train and validation only")
    selected_rows = tuning[tuning["selected"].astype(str).str.lower().eq("true")]
    check("one_selected_candidate", len(selected_rows) == 1, f"selected_rows={len(selected_rows)}")
    check("selected_candidate_consistent", len(selected_rows) == 1 and
          int(selected_rows.iloc[0]["candidate_id"]) == int(artifact["selected_candidate_id"]),
          f"artifact_candidate={artifact.get('selected_candidate_id')}")
    eligible = tuning[tuning["recall_constraint_met"].astype(str).str.lower().eq("true")]
    pool = eligible if len(eligible) else tuning
    expected = pool.sort_values(["validation_pr_auc", "validation_f1", "candidate_id"],
                                ascending=[False, False, True]).iloc[0]["candidate_id"]
    check("validation_only_ranking_reproduces", int(expected) == int(artifact["selected_candidate_id"]),
          f"expected={int(expected)}, selected={artifact['selected_candidate_id']}")
    forbidden = set(config["features"].get("forbidden_predictor_columns", []))
    leaked = [f for f in artifact["feature_names"] if f.split("__", 1)[0] in forbidden]
    check("no_target_leakage", not leaked, f"leaked={leaked}")
    check("feature_schema_matches", artifact["channel_names"] == config["features"]["channel_names"] and
          artifact["statistics"] == config["features"]["statistics"], "artifact feature order must match config")
    check("feature_importance_schema", list(importance.columns) == ["feature", "importance"] and
          set(importance["feature"]) == set(artifact["feature_names"]) and importance["importance"].is_monotonic_decreasing,
          "importance must include all features in descending order")
    check("feature_importance_sum", np.isclose(importance["importance"].sum(), 1.0, atol=1e-6),
          f"sum={importance['importance'].sum()}")
    prediction_required = {columns["patch_id"], columns["aoi_id"], columns["split"], columns["target"],
                           "probability", "prediction"}
    check("prediction_schema", prediction_required <= set(predictions),
          f"missing={sorted(prediction_required - set(predictions))}")
    check("one_prediction_per_manifest_row", len(predictions) == len(manifest) and
          predictions[columns["patch_id"]].is_unique and
          set(predictions[columns["patch_id"]]) == set(manifest[columns["patch_id"]]),
          f"predictions={len(predictions)}, manifest={len(manifest)}")
    threshold = float(artifact["threshold"])
    check("threshold_range", 0 <= threshold <= 1, f"threshold={threshold}")
    check("threshold_applied", predictions["prediction"].astype(np.int8).eq(
          predictions["probability"].ge(threshold).astype(np.int8)).all(), "saved threshold must reproduce predictions")
    differences = {}
    for split in ("train", "validation", "test"):
        part = predictions[predictions[columns["split"]].eq(split)]
        y = part[columns["target"]].astype(np.int8)
        p = part["probability"]
        pred = part["prediction"].astype(np.int8)
        values = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
                  "precision": precision_score(y, pred, zero_division=0),
                  "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}
        for metric, value in values.items():
            differences[f"{split}.{metric}"] = abs(float(value) - float(report["metrics_by_split"][split][metric]))
    check("reported_metrics_reproduce", max(differences.values(), default=0) <= 1e-12,
          f"maximum_absolute_difference={max(differences.values(), default=0):.3g}")
    baseline = json.loads(resolve(root, config["inputs"]["baseline_metrics"]).read_text(encoding="utf-8"))
    delta_ok = all(np.isclose(report["baseline_comparison_delta"][split][metric],
                              report["metrics_by_split"][split][metric] - baseline["metrics_by_split"][split][metric], atol=1e-12)
                   for split in ("validation", "test") for metric in ("roc_auc", "pr_auc", "precision", "recall", "f1"))
    check("baseline_comparison_reproduces", delta_ok, "all deltas must equal improved minus baseline")
    gate_ok = all(bool(g["passed"]) == (float(g["value"]) >= float(g["minimum"]))
                  for g in report["quality_gates"].values())
    check("quality_gate_reporting", gate_ok, f"status={report['quality_gate_status']}")
    result = {"task": "3.11", "status": "PASSED" if not errors else "FAILED",
              "quality_gate_status": report["quality_gate_status"], "checks": checks, "errors": errors}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Artifact validation: {result['status']}")
    print(f"Model quality gates: {result['quality_gate_status']}")
    print(f"Validation summary: {summary_path}")
    print(f"Task 3.11 validation: {result['status']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
