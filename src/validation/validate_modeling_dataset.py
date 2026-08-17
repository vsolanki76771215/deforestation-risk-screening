"""Validate Task 3.9 modeling data, spatial isolation, and reproducibility metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/modeling_dataset.yaml")
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    with (root / args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    columns = config["columns"]
    rules = config["validation"]
    split_cfg = config["split"]
    manifest_path = root / config["outputs"]["dataset_manifest"]
    output_path = root / config["outputs"]["validation_summary"]

    checks: list[dict] = []
    errors: list[str] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Modeling manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    required = [
        columns["patch_id"], columns["feature_patch_path"], columns["label_patch_path"],
        columns["aoi_id"], columns["aoi_role"], columns["row"], columns["col"],
        columns["patch_size"], config["label"]["source_column"], config["label"]["output_column"],
        "spatial_block_row", "spatial_block_col", "split",
    ]
    missing = [name for name in required if name not in frame.columns]
    check("required_columns", not missing, f"missing={missing}")
    if missing:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"task": "3.9", "status": "FAILED", "checks": checks}, indent=2) + "\n")
        print(f"Validation summary: {output_path}")
        print("Task 3.9 validation: FAILED")
        return 1

    check("non_empty", not frame.empty, f"rows={len(frame)}")
    patch_id_col = columns["patch_id"]
    check("unique_patch_id", frame[patch_id_col].is_unique, f"duplicates={frame[patch_id_col].duplicated().sum()}")
    check("no_null_required_values", not frame[required].isna().any().any(), "required columns contain no nulls")

    actual_splits = set(frame["split"].astype(str))
    allowed = set(rules["allowed_splits"])
    check("allowed_splits", actual_splits <= allowed, f"actual={sorted(actual_splits)}")
    if rules.get("require_all_splits", True):
        check("all_splits_present", actual_splits == allowed, f"actual={sorted(actual_splits)}")

    aoi_col = columns["aoi_id"]
    role_col = columns["aoi_role"]

    test_aois = set(split_cfg.get("held_out_test_aois", []))
    test_role = str(split_cfg["held_out_test_role"])

    final_holdout_aois = set(split_cfg.get("final_holdout_aois", []))
    final_holdout_role = str(
        split_cfg.get("final_holdout_role", "final_geographic_holdout")
    )

    expected_test = (
        frame[aoi_col].isin(test_aois)
        | frame[role_col].eq(test_role)
    )

    expected_final_holdout = (
        frame[aoi_col].isin(final_holdout_aois)
        | frame[role_col].eq(final_holdout_role)
    )

    development_rows = ~(expected_test | expected_final_holdout)

    check(
        "held_out_rows_are_test",
        frame.loc[expected_test, "split"].eq("test").all(),
        "diagnostic geographic-holdout rows must be test",
    )

    check(
        "final_holdout_rows_are_final_holdout",
        frame.loc[expected_final_holdout, "split"]
        .eq("final_holdout")
        .all(),
        "final geographic-holdout rows must be final_holdout",
    )

    check(
        "development_rows_are_train_or_validation",
        frame.loc[development_rows, "split"]
        .isin(["train", "validation"])
        .all(),
        "development AOIs must be assigned only to train or validation",
    )

    check(
        "test_and_final_holdout_are_disjoint",
        not (expected_test & expected_final_holdout).any(),
        "no row may belong to both test and final_holdout",
    )

    development = frame[frame["split"].isin(["train", "validation"])]
    block_split_counts = development.groupby(
        [aoi_col, "spatial_block_row", "spatial_block_col"]
    )["split"].nunique()
    check("spatial_blocks_do_not_cross_splits", block_split_counts.le(1).all(), f"violating_blocks={int(block_split_counts.gt(1).sum())}")
    if split_cfg.get("drop_block_boundary_patches", True):
        check(
            "development_patches_inside_blocks",
            development["fully_inside_spatial_block"].astype(bool).all(),
            "all train/validation patches must fit fully within their assigned block",
        )

    label_cfg = config["label"]
    label_col = label_cfg["output_column"]
    source = pd.to_numeric(frame[label_cfg["source_column"]], errors="coerce")
    expected_label = source.gt(float(label_cfg["positive_when_greater_than"])).astype("int8")
    actual_label = pd.to_numeric(frame[label_col], errors="coerce")
    label_matches = actual_label.notna().all() and actual_label.astype("int8").eq(expected_label).all()
    check("binary_label_rule", label_matches, "derived label must match configured loss-fraction rule")
    if rules.get("require_both_classes_in_each_split", True):
        class_counts = frame.groupby("split")[label_col].nunique()
        bad = class_counts[class_counts < 2].to_dict()
        check("both_classes_in_each_split", not bad, f"splits_with_fewer_than_2_classes={bad}")

    if rules.get("require_existing_patch_files", True):
        paths = []
        for path_column in (columns["feature_patch_path"], columns["label_patch_path"]):
            paths.extend(Path(p) if Path(p).is_absolute() else root / p for p in frame[path_column].astype(str))
        missing_paths = [str(p) for p in paths if not p.is_file()]
        check("patch_files_exist", not missing_paths, f"feature/label files missing={len(missing_paths)} examples={missing_paths[:5]}")
        if not missing_paths and rules.get("check_npz_readable", True):
            unreadable = []
            for path in paths:
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        if not archive.files:
                            raise ValueError("empty archive")
                except Exception as exc:  # report corrupt input without hiding its path
                    unreadable.append(f"{path}: {exc}")
            check("npz_files_readable", not unreadable, f"unreadable={len(unreadable)} examples={unreadable[:3]}")

    counts = {str(k): int(v) for k, v in frame["split"].value_counts().sort_index().items()}
    result = {
        "task": "3.9",
        "status": "PASSED" if not errors else "FAILED",
        "manifest": config["outputs"]["dataset_manifest"],
        "row_count": int(len(frame)),
        "counts_by_split": counts,
        "checks": checks,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for split_name, count in counts.items():
        print(f"{split_name}: {count} patches")
    print(f"Validation summary: {output_path}")
    print(f"Task 3.9 validation: {result['status']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())