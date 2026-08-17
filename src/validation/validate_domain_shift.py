"""Validate Task 3.12 diagnostic artifacts and leakage safeguards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/domain_shift_diagnostics.yaml")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = resolve(root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary_path = resolve(root, config["outputs"]["validation_summary"])
    checks, errors = [], []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    inputs = {k: resolve(root, v) for k, v in config["inputs"].items()}
    outputs = {k: resolve(root, v) for k, v in config["outputs"].items() if k != "validation_summary"}
    missing = [str(path) for path in [*inputs.values(), *outputs.values()] if not path.is_file()]
    check("required_files_exist", not missing, f"missing={missing}")
    if missing:
        result = {"task": "3.12", "status": "FAILED", "checks": checks, "errors": errors}
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Validation summary: {summary_path}"); print("Task 3.12 validation: FAILED")
        return 1

    report = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    split_profile, aoi_profile = pd.read_csv(outputs["split_profile"]), pd.read_csv(outputs["aoi_profile"])
    shift, errors_frame = pd.read_csv(outputs["feature_shift"]), pd.read_csv(outputs["error_analysis"])
    spatial = pd.read_csv(outputs["spatial_error_grid"])
    recommendations = json.loads(outputs["recommendations"].read_text(encoding="utf-8"))
    manifest = pd.read_csv(inputs["modeling_manifest"])
    predictions = pd.read_csv(inputs["improved_predictions"])
    columns, diag = config["columns"], config["diagnostics"]

    check("diagnostic_only_protocol", report.get("diagnostic_only") is True and
          report.get("model_retrained") is False and report.get("test_used_for_training_or_selection") is False,
          "Task 3.12 must not train, select, or tune a model")
    hash_ok = report.get("config_sha256") == sha256(config_path) and all(
        report.get("input_sha256", {}).get(name) == sha256(path) for name, path in inputs.items())
    check("input_integrity", hash_ok, "all recorded SHA-256 values must reproduce")
    expected_splits = {diag["reference_split"], *diag["comparison_splits"]}
    check("split_profile_complete", set(split_profile[columns["split"]]) == expected_splits and
          int(split_profile["patch_count"].sum()) == len(manifest),
          f"profile_rows={len(split_profile)}, profiled_patches={split_profile['patch_count'].sum()}")
    manifest_aoi_pairs = set(map(tuple, manifest[[columns["aoi_id"], columns["split"]]].drop_duplicates().to_numpy()))
    profile_aoi_pairs = set(map(tuple, aoi_profile[[columns["aoi_id"], columns["split"]]].to_numpy()))
    check("aoi_profile_complete", manifest_aoi_pairs == profile_aoi_pairs and
          int(aoi_profile["patch_count"].sum()) == len(manifest), "every AOI/split pair must be represented")

    expected_features = set(pd.read_csv(inputs["feature_importance"])["feature"])
    expected_shift_rows = len(expected_features) * len(diag["comparison_splits"])
    required_shift = {"feature", "comparison_split", "standardized_mean_difference",
                      "population_stability_index", "jensen_shannon_divergence", "model_importance",
                      "importance_weighted_smd"}
    check("feature_shift_schema", required_shift <= set(shift) and len(shift) == expected_shift_rows and
          set(shift["feature"]) == expected_features, f"rows={len(shift)}, expected={expected_shift_rows}")
    values_nonnegative = (shift[["standardized_mean_difference", "population_stability_index",
                                 "jensen_shannon_divergence", "model_importance",
                                 "importance_weighted_smd"]] >= -1e-12).all().all()
    check("feature_shift_values", values_nonnegative and np.isfinite(shift.select_dtypes("number")).all().all(),
          "shift metrics must be finite and nonnegative")
    weighted_ok = np.allclose(shift["importance_weighted_smd"],
                              shift["model_importance"] * shift["standardized_mean_difference"], atol=1e-12)
    check("weighted_shift_reproduces", weighted_ok, "weighted SMD must equal importance times SMD")

    outcome_totals = errors_frame.groupby(columns["split"])["patch_count"].sum().to_dict()
    manifest_totals = manifest.groupby(columns["split"]).size().to_dict()
    check("error_partition_complete", outcome_totals == manifest_totals,
          f"error_totals={outcome_totals}, manifest_totals={manifest_totals}")
    spatial_totals = spatial.groupby(columns["split"])["patch_count"].sum().to_dict()
    check("spatial_partition_complete", spatial_totals == manifest_totals,
          f"spatial_totals={spatial_totals}, manifest_totals={manifest_totals}")
    check("spatial_error_arithmetic", np.allclose(spatial["error_rate"],
          (spatial["false_positives"] + spatial["false_negatives"]) / spatial["patch_count"]),
          "error_rate arithmetic must reproduce")

    merged = manifest[[columns["patch_id"], columns["split"], columns["target"]]].merge(
        predictions[[columns["patch_id"], "prediction"]], on=columns["patch_id"], validate="one_to_one")
    expected_counts = merged.assign(error_type=np.select(
        [(merged[columns["target"]] == 0) & (merged["prediction"] == 0),
         (merged[columns["target"]] == 0) & (merged["prediction"] == 1),
         (merged[columns["target"]] == 1) & (merged["prediction"] == 0)],
        ["true_negative", "false_positive", "false_negative"], default="true_positive")).groupby(
            [columns["split"], "error_type"]).size().to_dict()
    actual_counts = errors_frame.groupby([columns["split"], "error_type"])["patch_count"].sum().to_dict()
    check("error_counts_reproduce", expected_counts == actual_counts,
          f"expected={expected_counts}, actual={actual_counts}")
    check("recommendations_preserve_holdout", all(item.get("uses_test_for_training") is False
          for item in recommendations.get("priority", [])) and len(recommendations.get("prohibited", [])) >= 2,
          "recommendations must preserve Tambopata as holdout")
    check("plots_nonempty", outputs["feature_shift_plot"].stat().st_size > 1000 and
          outputs["error_plot"].stat().st_size > 1000, "both plots must contain rendered output")

    result = {"task": "3.12", "status": "PASSED" if not errors else "FAILED",
              "diagnostic_status": report.get("status"), "model_quality_gate_status": report.get("model_quality_gate_status"),
              "checks": checks, "errors": errors}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Artifact validation: {result['status']}")
    print(f"Model quality gates (carried from Task 3.11): {result['model_quality_gate_status']}")
    print(f"Validation summary: {summary_path}")
    print(f"Task 3.12 validation: {result['status']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
