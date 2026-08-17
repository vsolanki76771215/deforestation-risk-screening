"""Diagnose geographic domain shift and model errors without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def summarize_patch(array, statistics):
    reducers = {
        "mean": lambda x: np.mean(x, axis=(1, 2)), "std": lambda x: np.std(x, axis=(1, 2)),
        "min": lambda x: np.min(x, axis=(1, 2)), "p10": lambda x: np.percentile(x, 10, axis=(1, 2)),
        "p25": lambda x: np.percentile(x, 25, axis=(1, 2)), "median": lambda x: np.median(x, axis=(1, 2)),
        "p75": lambda x: np.percentile(x, 75, axis=(1, 2)), "p90": lambda x: np.percentile(x, 90, axis=(1, 2)),
        "max": lambda x: np.max(x, axis=(1, 2)),
    }
    unknown = set(statistics) - set(reducers)
    if unknown:
        raise ValueError(f"Unsupported statistics: {sorted(unknown)}")
    return np.stack([reducers[s](array) for s in statistics], axis=1).reshape(-1).astype(np.float32)


def load_features(frame, root, artifact, path_column, progress_every):
    channels, statistics = artifact["channel_names"], artifact["statistics"]
    names = artifact["feature_names"]
    expected = (len(channels), int(artifact["expected_patch_size"]), int(artifact["expected_patch_size"]))
    matrix = np.empty((len(frame), len(names)), dtype=np.float32)
    for index, value in enumerate(frame[path_column].astype(str), 1):
        path = resolve(root, value)
        with np.load(path, allow_pickle=False) as archive:
            array = archive[artifact["archive_key"]]
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"Invalid feature patch {path}: shape={array.shape}")
        matrix[index - 1] = summarize_patch(array, statistics)
        if progress_every > 0 and (index % progress_every == 0 or index == len(frame)):
            print(f"Diagnostic feature extraction: {index}/{len(frame)} ({100*index/len(frame):.1f}%)", flush=True)
    return matrix


def proportions(values, edges, epsilon):
    counts = np.histogram(values, bins=edges)[0].astype(float) + epsilon
    return counts / counts.sum()


def shift_scores(reference, comparison, bins, epsilon):
    pooled = np.concatenate([reference, comparison])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0, 0.0, 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p, q = proportions(reference, edges, epsilon), proportions(comparison, edges, epsilon)
    pooled_std = np.sqrt((np.var(reference) + np.var(comparison)) / 2)
    smd = abs(np.mean(comparison) - np.mean(reference)) / pooled_std if pooled_std > epsilon else 0.0
    psi = np.sum((q - p) * np.log(q / p))
    midpoint = (p + q) / 2
    jsd = 0.5 * np.sum(p * np.log2(p / midpoint)) + 0.5 * np.sum(q * np.log2(q / midpoint))
    return float(smd), float(psi), float(jsd)


def profile(group, target, probability, prediction):
    y, pred = group[target].astype(int), group[prediction].astype(int)
    tp, tn = ((y == 1) & (pred == 1)).sum(), ((y == 0) & (pred == 0)).sum()
    fp, fn = ((y == 0) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum()
    return pd.Series({"patch_count": len(group), "positive_count": int(y.sum()),
                      "positive_rate": float(y.mean()), "mean_probability": float(group[probability].mean()),
                      "predicted_positive_rate": float(pred.mean()), "tn": int(tn), "fp": int(fp),
                      "fn": int(fn), "tp": int(tp)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/domain_shift_diagnostics.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = resolve(root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = {k: resolve(root, v) for k, v in config["inputs"].items()}
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Task 3.12 inputs: {missing}")
    outputs = {k: resolve(root, v) for k, v in config["outputs"].items() if k != "validation_summary"}
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Task 3.12 output exists; pass --overwrite: {existing}")
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    columns, diag = config["columns"], config["diagnostics"]
    manifest, predictions = pd.read_csv(inputs["modeling_manifest"]), pd.read_csv(inputs["improved_predictions"])
    artifact = joblib.load(inputs["improved_model"])
    key = columns["patch_id"]
    if not manifest[key].is_unique or not predictions[key].is_unique:
        raise ValueError("Patch IDs must be unique")
    pred_cols = [key, "probability", "prediction"]
    frame = manifest.merge(predictions[pred_cols], on=key, validate="one_to_one")
    if len(frame) != len(manifest):
        raise ValueError("Predictions do not cover every manifest patch")
    forbidden = set(diag["forbidden_predictor_columns"])
    leaked = [name for name in artifact["feature_names"] if name.split("__", 1)[0] in forbidden]
    if leaked or artifact.get("test_used_for_selection") is not False:
        raise ValueError(f"Leakage safeguard failed; leaked={leaked}")

    matrix = load_features(frame, root, artifact, columns["feature_patch_path"], args.progress_every)
    reference = diag["reference_split"]
    split_values = frame[columns["split"]].astype(str).to_numpy()
    reference_mask = split_values == reference
    importance = pd.read_csv(inputs["feature_importance"]).set_index("feature")["importance"].to_dict()
    rows = []
    for comparison in diag["comparison_splits"]:
        comparison_mask = split_values == comparison
        for index, name in enumerate(artifact["feature_names"]):
            smd, psi, jsd = shift_scores(matrix[reference_mask, index], matrix[comparison_mask, index],
                                         int(diag["histogram_bins"]), float(diag["epsilon"]))
            rows.append({"reference_split": reference, "comparison_split": comparison, "feature": name,
                         "channel": name.split("__", 1)[0], "statistic": name.split("__", 1)[1],
                         "reference_mean": float(matrix[reference_mask, index].mean()),
                         "comparison_mean": float(matrix[comparison_mask, index].mean()),
                         "standardized_mean_difference": smd, "population_stability_index": psi,
                         "jensen_shannon_divergence": jsd, "model_importance": float(importance.get(name, 0.0)),
                         "importance_weighted_smd": float(importance.get(name, 0.0)) * smd})
    shifts = pd.DataFrame(rows).sort_values(["comparison_split", "importance_weighted_smd"], ascending=[True, False])
    shifts.to_csv(outputs["feature_shift"], index=False, lineterminator="\n")

    target, split, aoi = columns["target"], columns["split"], columns["aoi_id"]
    split_profile = frame.groupby(split, sort=True).apply(profile, target=target, probability="probability",
                                                           prediction="prediction", include_groups=False).reset_index()
    aoi_profile = frame.groupby([aoi, split], sort=True).apply(profile, target=target, probability="probability",
                                                               prediction="prediction", include_groups=False).reset_index()
    split_profile.to_csv(outputs["split_profile"], index=False, lineterminator="\n")
    aoi_profile.to_csv(outputs["aoi_profile"], index=False, lineterminator="\n")

    frame["error_type"] = np.select([(frame[target] == 0) & (frame["prediction"] == 0),
                                      (frame[target] == 0) & (frame["prediction"] == 1),
                                      (frame[target] == 1) & (frame["prediction"] == 0)],
                                     ["true_negative", "false_positive", "false_negative"], default="true_positive")
    errors = frame.groupby([split, aoi, "error_type"], sort=True).agg(
        patch_count=(key, "size"), mean_probability=("probability", "mean"),
        mean_loss_fraction=("loss_fraction", "mean") if "loss_fraction" in frame else (target, "mean")).reset_index()
    errors.to_csv(outputs["error_analysis"], index=False, lineterminator="\n")
    grid_size = int(diag["spatial_grid_size_pixels"])
    frame["grid_row"] = (frame[columns["row_off"]].astype(int) // grid_size).astype(int)
    frame["grid_col"] = (frame[columns["col_off"]].astype(int) // grid_size).astype(int)
    spatial = frame.groupby([split, aoi, "grid_row", "grid_col"], sort=True).agg(
        patch_count=(key, "size"), positive_rate=(target, "mean"), mean_probability=("probability", "mean"),
        false_positives=("error_type", lambda x: int((x == "false_positive").sum())),
        false_negatives=("error_type", lambda x: int((x == "false_negative").sum()))).reset_index()
    spatial["error_rate"] = (spatial["false_positives"] + spatial["false_negatives"]) / spatial["patch_count"]
    spatial.to_csv(outputs["spatial_error_grid"], index=False, lineterminator="\n")

    test_shifts = shifts[shifts["comparison_split"].eq("test")]
    severe = test_shifts[(test_shifts["standardized_mean_difference"] >= float(diag["severe_smd"])) |
                         (test_shifts["population_stability_index"] >= float(diag["severe_psi"])) |
                         (test_shifts["jensen_shannon_divergence"] >= float(diag["severe_jsd"]))]
    prevalence = split_profile.set_index(split)["positive_rate"]
    recommendations = {
        "priority": [
            {"rank": 1, "action": "Review top importance-weighted shifted channels and composite normalization", "uses_test_for_training": False},
            {"rank": 2, "action": "Inspect high-error spatial cells for label, cloud, boundary, and mining-context artifacts", "uses_test_for_training": False},
            {"rank": 3, "action": "Add geographically diverse development AOIs, then rebuild spatial train/validation splits", "uses_test_for_training": False},
            {"rank": 4, "action": "Evaluate spatial/contextual models and class-prior calibration on new validation AOIs", "uses_test_for_training": False}],
        "prohibited": ["Do not tune thresholds or hyperparameters on Tambopata", "Do not move Tambopata patches into training"],
    }
    outputs["recommendations"].write_text(json.dumps(recommendations, indent=2) + "\n", encoding="utf-8")

    top = test_shifts.nlargest(int(diag["top_features"]), "importance_weighted_smd").sort_values("importance_weighted_smd")
    plt.figure(figsize=(10, 7)); plt.barh(top["feature"], top["importance_weighted_smd"], color="#c44e52")
    plt.xlabel("Importance-weighted standardized mean difference"); plt.title("Top train-to-test feature shifts")
    plt.tight_layout(); plt.savefig(outputs["feature_shift_plot"], dpi=160); plt.close()
    test_errors = frame[frame[split].eq("test")]["error_type"].value_counts().reindex(
        ["true_negative", "false_positive", "false_negative", "true_positive"], fill_value=0)
    plt.figure(figsize=(8, 5)); test_errors.plot.bar(color=["#4c72b0", "#dd8452", "#c44e52", "#55a868"])
    plt.ylabel("Patch count"); plt.title("Held-out test prediction outcomes"); plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); plt.savefig(outputs["error_plot"], dpi=160); plt.close()

    metrics = json.loads(inputs["improved_metrics"].read_text(encoding="utf-8"))
    summary = {"task": "3.12", "status": "COMPLETED", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "diagnostic_only": True, "model_retrained": False, "test_used_for_training_or_selection": False,
               "reference_split": reference, "comparison_splits": diag["comparison_splits"],
               "model_quality_gate_status": metrics["quality_gate_status"],
               "train_positive_rate": float(prevalence[reference]), "test_positive_rate": float(prevalence["test"]),
               "positive_rate_ratio_test_to_train": float(prevalence["test"] / prevalence[reference]),
               "severely_shifted_test_feature_count": int(len(severe)),
               "top_shifted_test_features": test_shifts.nlargest(10, "importance_weighted_smd")[
                   ["feature", "standardized_mean_difference", "population_stability_index",
                    "jensen_shannon_divergence", "model_importance", "importance_weighted_smd"]].to_dict("records"),
               "input_sha256": {k: sha256(v) for k, v in inputs.items()},
               "config_sha256": sha256(config_path), "artifacts": config["outputs"]}
    outputs["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Severely shifted test features: {len(severe)}/{len(test_shifts)}")
    print(f"Train/test positive rate: {prevalence[reference]:.4f}/{prevalence['test']:.4f}")
    print(f"Summary: {outputs['summary']}")
    print("Task 3.12 domain-shift diagnosis: COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
