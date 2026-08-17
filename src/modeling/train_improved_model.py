"""Tune and evaluate the Task 3.11 tree-ensemble model without test leakage."""

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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/improved_model.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_names(channels, statistics):
    return [f"{channel}__{stat}" for channel in channels for stat in statistics]


def summarize_patch(array, statistics):
    reducers = {
        "mean": lambda x: np.mean(x, axis=(1, 2)),
        "std": lambda x: np.std(x, axis=(1, 2)),
        "min": lambda x: np.min(x, axis=(1, 2)),
        "p10": lambda x: np.percentile(x, 10, axis=(1, 2)),
        "p25": lambda x: np.percentile(x, 25, axis=(1, 2)),
        "median": lambda x: np.median(x, axis=(1, 2)),
        "p75": lambda x: np.percentile(x, 75, axis=(1, 2)),
        "p90": lambda x: np.percentile(x, 90, axis=(1, 2)),
        "max": lambda x: np.max(x, axis=(1, 2)),
    }
    unknown = sorted(set(statistics) - set(reducers))
    if unknown:
        raise ValueError(f"Unsupported feature statistics: {unknown}")
    return np.stack([reducers[s](array) for s in statistics], axis=1).reshape(-1).astype(np.float32)


def load_features(frame, root, config, path_column, progress_every):
    channels = config["channel_names"]
    statistics = config["statistics"]
    matrix = np.empty((len(frame), len(channels) * len(statistics)), dtype=np.float32)
    expected = (len(channels), int(config["expected_patch_size"]), int(config["expected_patch_size"]))
    for index, value in enumerate(frame[path_column].astype(str), 1):
        path = resolve(root, value)
        with np.load(path, allow_pickle=False) as archive:
            if config["archive_key"] not in archive:
                raise KeyError(f"{path} has no {config['archive_key']!r} array")
            array = archive[config["archive_key"]]
        if array.shape != expected:
            raise ValueError(f"{path}: expected {expected}, found {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: non-finite feature values")
        matrix[index - 1] = summarize_patch(array, statistics)
        if progress_every > 0 and (index % progress_every == 0 or index == len(frame)):
            print(f"Feature extraction: {index}/{len(frame)} ({100 * index / len(frame):.1f}%)", flush=True)
    return matrix


def threshold_search(y, probability, minimum_recall, points):
    if points < 2:
        raise ValueError("threshold_grid_points must be at least 2")
    rows = []
    for threshold in np.linspace(0.0, 1.0, points):
        predicted = probability >= threshold
        rows.append((threshold,
                     recall_score(y, predicted, zero_division=0),
                     precision_score(y, predicted, zero_division=0),
                     f1_score(y, predicted, zero_division=0)))
    eligible = [row for row in rows if row[1] >= minimum_recall]
    threshold, recall, precision, f1 = max(eligible or rows, key=lambda r: (r[3], r[2], r[0]))
    return {"threshold": float(threshold), "validation_recall": float(recall),
            "validation_precision": float(precision), "validation_f1": float(f1),
            "minimum_recall_constraint_met": bool(eligible)}


def calculate_metrics(y, probability, threshold):
    predicted = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"row_count": int(len(y)), "positive_count": int(y.sum()),
            "negative_count": int(len(y) - y.sum()), "threshold": float(threshold),
            "roc_auc": float(roc_auc_score(y, probability)),
            "pr_auc": float(average_precision_score(y, probability)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def build_model(candidate, seed, n_jobs):
    params = {k: v for k, v in candidate.items() if k != "model"}
    params.update(random_state=seed, n_jobs=n_jobs)
    if candidate["model"] == "ExtraTreesClassifier":
        return ExtraTreesClassifier(**params)
    if candidate["model"] == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    raise ValueError(f"Unsupported candidate model: {candidate['model']}")


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    config_path = resolve(root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = resolve(root, config["inputs"]["modeling_manifest"])
    baseline_path = resolve(root, config["inputs"]["baseline_metrics"])
    if not manifest_path.is_file() or not baseline_path.is_file():
        raise FileNotFoundError(f"Required input missing: manifest={manifest_path.is_file()}, baseline={baseline_path.is_file()}")
    outputs = {k: resolve(root, v) for k, v in config["outputs"].items() if k != "validation_summary"}
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Task 3.11 output exists; pass --overwrite to replace it: {existing}")

    frame = pd.read_csv(manifest_path)
    columns = config["columns"]
    required = [columns[k] for k in ("patch_id", "aoi_id", "feature_patch_path", "target", "split")]
    missing = sorted(set(required) - set(frame.columns))
    if missing or frame.empty or frame[columns["patch_id"]].duplicated().any():
        raise ValueError(f"Invalid modeling manifest; missing={missing}, rows={len(frame)}")
    if set(frame[columns["split"]].astype(str)) != {"train", "validation", "test"}:
        raise ValueError("Modeling manifest must contain exactly train, validation, and test")
    y = pd.to_numeric(frame[columns["target"]], errors="raise").astype(np.int8).to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Target must be binary")
    masks = {s: frame[columns["split"]].eq(s).to_numpy() for s in ("train", "validation", "test")}
    if any(np.unique(y[mask]).size != 2 for mask in masks.values()):
        raise ValueError("Every split must contain both classes")
    feature_cfg = config["features"]
    forbidden = set(feature_cfg.get("forbidden_predictor_columns", []))
    if forbidden.intersection(feature_cfg["channel_names"]):
        raise ValueError("Target-derived feature channel configured")
    names = feature_names(feature_cfg["channel_names"], feature_cfg["statistics"])
    matrix = load_features(frame, root, feature_cfg, columns["feature_patch_path"], args.progress_every)

    tune = config["tuning"]
    if tune.get("selection_metric") != "pr_auc":
        raise ValueError("Task 3.11 selection_metric must be pr_auc")
    candidates = tune["candidates"]
    if not candidates:
        raise ValueError("At least one tuning candidate is required")
    tuning_rows, fitted = [], []
    print(f"Tuning {len(candidates)} candidates using train -> validation only...", flush=True)
    for candidate_id, candidate in enumerate(candidates, 1):
        model = build_model(candidate, int(config["random_seed"]), int(tune["n_jobs"]))
        model.fit(matrix[masks["train"]], y[masks["train"]])
        probability = model.predict_proba(matrix[masks["validation"]])[:, 1]
        selected = threshold_search(y[masks["validation"]], probability,
                                    float(tune["minimum_recall"]), int(tune["threshold_grid_points"]))
        row = {"candidate_id": candidate_id, "model": candidate["model"],
               "parameters_json": json.dumps(candidate, sort_keys=True),
               "validation_roc_auc": roc_auc_score(y[masks["validation"]], probability),
               "validation_pr_auc": average_precision_score(y[masks["validation"]], probability),
               "threshold": selected["threshold"], "validation_precision": selected["validation_precision"],
               "validation_recall": selected["validation_recall"], "validation_f1": selected["validation_f1"],
               "recall_constraint_met": selected["minimum_recall_constraint_met"]}
        tuning_rows.append(row)
        fitted.append(model)
        print(f"Candidate {candidate_id}/{len(candidates)} {candidate['model']}: "
              f"val PR-AUC={row['validation_pr_auc']:.4f}, ROC-AUC={row['validation_roc_auc']:.4f}, "
              f"recall={row['validation_recall']:.4f}", flush=True)

    eligible_ids = [i for i, row in enumerate(tuning_rows) if row["recall_constraint_met"]]
    pool = eligible_ids or list(range(len(tuning_rows)))
    winner_index = max(pool, key=lambda i: (tuning_rows[i]["validation_pr_auc"],
                                            tuning_rows[i]["validation_f1"], -tuning_rows[i]["candidate_id"]))
    winner, model = tuning_rows[winner_index], fitted[winner_index]
    threshold = float(winner["threshold"])
    # Test is accessed only here, after model and threshold selection are final.
    probabilities = {s: model.predict_proba(matrix[mask])[:, 1] for s, mask in masks.items()}
    metrics_by_split = {s: calculate_metrics(y[masks[s]], probabilities[s], threshold) for s in masks}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    comparison = {s: {metric: metrics_by_split[s][metric] - baseline["metrics_by_split"][s][metric]
                      for metric in ("roc_auc", "pr_auc", "precision", "recall", "f1")}
                  for s in ("validation", "test")}
    gates_cfg = config["quality_gates"]
    gates = {
        "test_roc_auc": {"value": metrics_by_split["test"]["roc_auc"], "minimum": float(gates_cfg["test_roc_auc_min"])},
        "test_pr_auc": {"value": metrics_by_split["test"]["pr_auc"], "minimum": float(gates_cfg["test_pr_auc_min"])},
        "test_recall": {"value": metrics_by_split["test"]["recall"], "minimum": float(gates_cfg["test_recall_min"])},
    }
    for gate in gates.values():
        gate["passed"] = gate["value"] >= gate["minimum"]
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "threshold": threshold, "feature_names": names,
                "channel_names": feature_cfg["channel_names"], "statistics": feature_cfg["statistics"],
                "archive_key": feature_cfg["archive_key"], "expected_patch_size": feature_cfg["expected_patch_size"],
                "target_column": columns["target"], "random_seed": config["random_seed"],
                "selected_candidate_id": winner["candidate_id"], "selected_parameters": candidates[winner_index],
                "selection_used_splits": ["train", "validation"], "test_used_for_selection": False,
                "manifest_sha256": sha256(manifest_path), "config_sha256": sha256(config_path)}
    joblib.dump(artifact, outputs["model_artifact"], compress=3)
    parts = []
    for split, mask in masks.items():
        part = frame.loc[mask, [columns["patch_id"], columns["aoi_id"], columns["split"], columns["target"]]].copy()
        part["probability"] = probabilities[split]
        part["prediction"] = (probabilities[split] >= threshold).astype(np.int8)
        parts.append(part)
    pd.concat(parts, ignore_index=True).to_csv(outputs["predictions"], index=False, lineterminator="\n")
    tuning_frame = pd.DataFrame(tuning_rows)
    tuning_frame["selected"] = tuning_frame["candidate_id"].eq(winner["candidate_id"])
    tuning_frame.to_csv(outputs["tuning_results"], index=False, lineterminator="\n")
    pd.DataFrame({"feature": names, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(outputs["feature_importance"], index=False, lineterminator="\n")
    report = {"task": "3.11", "status": "COMPLETED", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection_protocol": {"fit_split": "train", "ranking_split": "validation",
                                     "test_evaluations_after_selection": 1, "test_used_for_selection": False},
              "selected_candidate": winner, "feature_count": len(names), "features": names,
              "metrics_by_split": metrics_by_split, "baseline_comparison_delta": comparison,
              "quality_gates": gates,
              "quality_gate_status": "PASSED" if all(g["passed"] for g in gates.values()) else "FAILED",
              "manifest_sha256": artifact["manifest_sha256"], "config_sha256": artifact["config_sha256"],
              "artifacts": {k: config["outputs"][k] for k in outputs},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                          "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}}
    outputs["metrics"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selected candidate: {winner['candidate_id']} ({winner['model']}); threshold={threshold:.3f}")
    for split in ("validation", "test"):
        item = metrics_by_split[split]
        print(f"{split}: ROC-AUC={item['roc_auc']:.4f}, PR-AUC={item['pr_auc']:.4f}, "
              f"precision={item['precision']:.4f}, recall={item['recall']:.4f}, F1={item['f1']:.4f}")
    print(f"Quality gates: {report['quality_gate_status']}")
    print(f"Metrics: {outputs['metrics']}")
    print("Task 3.11 improved model training: COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
