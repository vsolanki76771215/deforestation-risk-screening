"""Build the Task 3.9 modeling manifest and deterministic spatial splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/modeling_dataset.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def stable_unit_interval(seed: int, aoi_id: str, block_row: int, block_col: int) -> float:
    value = f"{seed}|{aoi_id}|{block_row}|{block_col}".encode("utf-8")
    digest = hashlib.sha256(value).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def normalize_path(value: object) -> str:
    return str(value).replace("\\", "/")


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    config = load_yaml(root / args.config)
    columns = config["columns"]
    split_cfg = config["split"]

    input_path = root / config["inputs"]["patch_manifest"]
    output_path = root / config["outputs"]["dataset_manifest"]
    summary_path = root / config["outputs"]["split_summary"]
    if not input_path.exists():
        raise FileNotFoundError(f"Patch manifest not found: {input_path}")
    if (output_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError("Task 3.9 output exists; pass --overwrite to replace it")

    frame = pd.read_csv(input_path)
    required = [
        columns["patch_id"], columns["feature_patch_path"], columns["label_patch_path"],
        columns["aoi_id"], columns["aoi_role"], columns["row"], columns["col"],
        columns["patch_size"], config["label"]["source_column"],
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Patch manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Patch manifest contains no rows")

    row_col = columns["row"]
    col_col = columns["col"]
    aoi_col = columns["aoi_id"]
    role_col = columns["aoi_role"]
    patch_id_col = columns["patch_id"]
    feature_path_col = columns["feature_patch_path"]
    label_path_col = columns["label_patch_path"]
    for name in (row_col, col_col):
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype("int64")
        if (frame[name] < 0).any():
            raise ValueError(f"Column {name} contains negative pixel coordinates")

    block_size = int(split_cfg["spatial_block_size_pixels"])
    patch_size = int(split_cfg["patch_size_pixels"])
    if block_size < patch_size:
        raise ValueError("spatial_block_size_pixels must be >= patch_size_pixels")

    if frame[patch_id_col].isna().any() or frame[patch_id_col].astype(str).str.strip().eq("").any():
        raise ValueError(f"Column {patch_id_col} contains blank patch IDs")
    if frame[patch_id_col].duplicated().any():
        examples = frame.loc[frame[patch_id_col].duplicated(False), patch_id_col].head().tolist()
        raise ValueError(f"Duplicate patch IDs found; examples: {examples}")
    for path_col in (feature_path_col, label_path_col):
        frame[path_col] = frame[path_col].map(normalize_path)

    manifest_patch_sizes = pd.to_numeric(frame[columns["patch_size"]], errors="raise").astype("int64")
    if not manifest_patch_sizes.eq(patch_size).all():
        found = sorted(manifest_patch_sizes.unique().tolist())
        raise ValueError(f"Manifest patch_size values {found} do not match configured value {patch_size}")

    label_cfg = config["label"]
    label_source = label_cfg["source_column"]
    label_col = label_cfg["output_column"]
    threshold = float(label_cfg["positive_when_greater_than"])
    label_values = pd.to_numeric(frame[label_source], errors="raise")
    if label_values.isna().any() or ((label_values < 0.0) | (label_values > 1.0)).any():
        raise ValueError(f"Column {label_source} must contain values from 0 through 1")
    frame[label_col] = label_values.gt(threshold).astype("int8")
    frame["spatial_block_row"] = frame[row_col] // block_size
    frame["spatial_block_col"] = frame[col_col] // block_size
    row_offset = frame[row_col] % block_size
    col_offset = frame[col_col] % block_size
    frame["fully_inside_spatial_block"] = (
        (row_offset + patch_size <= block_size)
        & (col_offset + patch_size <= block_size)
    )

    test_aois = set(split_cfg.get("held_out_test_aois", []))
    test_role = str(split_cfg["held_out_test_role"])
    final_holdout_aois = set(split_cfg.get("final_holdout_aois", []))
    final_holdout_role = str(
        split_cfg.get("final_holdout_role", "final_geographic_holdout")
    )

    is_test = frame[aoi_col].isin(test_aois) | frame[role_col].eq(test_role)
    is_final_holdout = (
        frame[aoi_col].isin(final_holdout_aois)
        | frame[role_col].eq(final_holdout_role)
    )
    overlapping_holdouts = is_test & is_final_holdout
    if overlapping_holdouts.any():
        examples = (
            frame.loc[overlapping_holdouts, [aoi_col, role_col]]
            .drop_duplicates()
            .head()
            .to_dict(orient="records")
        )
        raise ValueError(
            "Patches cannot belong to both test and final_holdout; "
            f"examples: {examples}"
        )

    frame["split"] = ""
    frame.loc[is_test, "split"] = "test"
    frame.loc[is_final_holdout, "split"] = "final_holdout"

    if split_cfg.get("drop_block_boundary_patches", True):
        # Holdouts are retained intact. Boundary filtering is used only to prevent
        # spatial leakage between development train and validation blocks.
        keep = is_test | is_final_holdout | frame["fully_inside_spatial_block"]
        dropped = frame.loc[~keep].copy()
        frame = frame.loc[keep].copy()
    else:
        dropped = frame.iloc[0:0].copy()

    development = frame["split"].eq("")

    validation_fraction = float(split_cfg["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    seed = int(config["random_seed"])

    block_keys = frame.loc[development, [aoi_col, "spatial_block_row", "spatial_block_col"]].drop_duplicates()
    block_keys["block_score"] = block_keys.apply(
        lambda r: stable_unit_interval(
            seed, str(r[aoi_col]), int(r["spatial_block_row"]), int(r["spatial_block_col"])
        ), axis=1,
    )
    block_keys["development_split"] = block_keys["block_score"].lt(validation_fraction).map(
        {True: "validation", False: "train"}
    )
    frame = frame.merge(
        block_keys,
        on=[aoi_col, "spatial_block_row", "spatial_block_col"],
        how="left",
        validate="many_to_one",
    )
    # merge may rebuild the DataFrame index, so derive this mask again afterward.
    development_rows = frame["split"].eq("")
    frame.loc[development_rows, "split"] = frame.loc[
        development_rows, "development_split"
    ]
    frame.drop(columns=["development_split"], inplace=True)

    if frame["split"].isna().any() or frame["split"].eq("").any():
        raise RuntimeError("One or more patches were not assigned to a split")

    # Stable sorting makes reruns byte-for-byte reproducible while preserving Task 3.8 IDs.
    frame.sort_values(["split", aoi_col, row_col, col_col, patch_id_col], kind="mergesort", inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, lineterminator="\n")

    counts = frame.groupby(["split", aoi_col], dropna=False).size().rename("patches").reset_index()
    class_counts = frame.groupby(["split", label_col], dropna=False).size().rename("patches").reset_index()
    summary = {
        "task": "3.9",
        "status": "COMPLETED",
        "random_seed": seed,
        "input_manifest": config["inputs"]["patch_manifest"],
        "output_manifest": config["outputs"]["dataset_manifest"],
        "input_patch_count": int(len(frame) + len(dropped)),
        "output_patch_count": int(len(frame)),
        "dropped_block_boundary_patch_count": int(len(dropped)),
        "label_rule": f"{label_col} = ({label_source} > {threshold})",
        "counts_by_split": {str(k): int(v) for k, v in frame["split"].value_counts().sort_index().items()},
        "counts_by_split_and_aoi": counts.to_dict(orient="records"),
        "class_counts_by_split": class_counts.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for split_name, count in summary["counts_by_split"].items():
        print(f"{split_name}: {count} patches")
    print(f"Dropped at spatial-block boundaries: {len(dropped)} patches")
    print(f"Modeling manifest: {output_path}")
    print("Task 3.9 dataset build: COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
