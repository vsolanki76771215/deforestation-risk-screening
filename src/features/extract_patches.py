#!/usr/bin/env python3
"""Task 3.8: deterministically extract paired feature/label patches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window


FEATURE_PATH_COLUMNS = ("feature_path", "features_path", "feature_stack_path")
VALID_PATH_COLUMNS = ("valid_mask_path", "model_valid_mask_path")
LABEL_PATH_COLUMNS = (
    "label_path",
    "label_raster_path",
    "output_path",
)


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: manifest has no header")

        rows = []
        for raw_row in reader:
            row = {
                key.strip().lower(): value.strip() if value else ""
                for key, value in raw_row.items()
                if key is not None
            }

            # Task 3.6 uses "aoi"; Task 3.7 uses "aoi_id".
            if "aoi_id" not in row and "aoi" in row:
                row["aoi_id"] = row["aoi"]

            rows.append(row)

    if rows and "aoi_id" not in rows[0]:
        raise ValueError(
            f"{path}: requires 'aoi_id' or 'aoi'. "
            f"Available columns: {list(rows[0])}"
        )

    return rows


def pick(row: dict[str, str], names: tuple[str, ...], description: str) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    raise ValueError(f"Missing {description}; expected one of {names}")


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def same_grid(left: rasterio.DatasetReader, right: rasterio.DatasetReader) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and left.crs == right.crs
        and left.transform.almost_equals(right.transform)
    )


def candidate_windows(width: int, height: int, size: int, stride: int) -> list[tuple[int, int]]:
    if width < size or height < size:
        return []
    return [(row, col) for row in range(0, height - size + 1, stride)
            for col in range(0, width - size + 1, stride)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "patch_id", "aoi_id", "aoi_role", "row_off", "col_off", "patch_size",
        "feature_patch_path", "label_patch_path", "valid_fraction", "positive_pixels",
        "negative_pixels", "loss_fraction", "feature_sha256", "label_sha256",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/patch_extraction.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    config = yaml.safe_load(resolve(root, args.config).read_text(encoding="utf-8"))
    settings = config["patches"]
    size = int(settings["patch_size"])
    stride = int(settings["stride"])
    seed = int(settings["random_seed"])
    min_valid = float(settings["min_valid_fraction"])
    max_per_aoi = settings.get("max_patches_per_aoi")
    max_per_aoi = None if max_per_aoi in (None, "") else int(max_per_aoi)
    if size <= 0 or stride <= 0 or not 0.0 <= min_valid <= 1.0:
        raise ValueError("patch_size/stride must be positive and min_valid_fraction must be in [0,1]")

    feature_manifest = resolve(root, config["inputs"]["feature_manifest"])
    label_manifest = resolve(root, config["inputs"]["label_manifest"])
    output_root = resolve(root, settings["output_root"])
    manifest_path = resolve(root, settings["manifest"])
    summary_path = resolve(root, settings["summary"])

    feature_rows = {row["aoi_id"]: row for row in read_rows(feature_manifest)}
    label_rows = {row["aoi_id"]: row for row in read_rows(label_manifest)}
    if len(feature_rows) != len(read_rows(feature_manifest)) or len(label_rows) != len(read_rows(label_manifest)):
        raise ValueError("Each input manifest must contain exactly one row per aoi_id")
    if set(feature_rows) != set(label_rows):
        raise ValueError("Feature and label manifests contain different AOI sets")

    records: list[dict[str, object]] = []
    aoi_summaries: list[dict[str, object]] = []
    for aoi_index, aoi_id in enumerate(sorted(feature_rows)):
        feature_row, label_row = feature_rows[aoi_id], label_rows[aoi_id]
        role = feature_row.get("aoi_role", label_row.get("aoi_role", "")).strip()
        if role != label_row.get("aoi_role", role).strip():
            raise ValueError(f"{aoi_id}: conflicting AOI roles")
        feature_path = resolve(root, pick(feature_row, FEATURE_PATH_COLUMNS, "feature stack path"))
        valid_path = resolve(root, pick(feature_row, VALID_PATH_COLUMNS, "valid-mask path"))
        label_path = resolve(root, pick(label_row, LABEL_PATH_COLUMNS, "label path"),)
        for path in (feature_path, valid_path, label_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        with rasterio.open(feature_path) as features, rasterio.open(valid_path) as valid, rasterio.open(label_path) as labels:
            if not same_grid(features, valid) or not same_grid(features, labels):
                raise ValueError(f"{aoi_id}: feature, valid-mask, and label grids are not aligned")
            if features.count != 11 or valid.count != 1 or labels.count != 1:
                raise ValueError(f"{aoi_id}: expected 11 feature bands and one-band masks/labels")

            candidates: list[tuple[int, int, float]] = []
            for row_off, col_off in candidate_windows(features.width, features.height, size, stride):
                window = Window(col_off, row_off, size, size)
                valid_data = valid.read(1, window=window)
                label_data = labels.read(1, window=window)
                eligible = (valid_data == 1) & np.isin(label_data, (0, 1))
                fraction = float(eligible.mean())
                if fraction >= min_valid:
                    candidates.append((row_off, col_off, fraction))

            # AOI-specific RNG makes an AOI reproducible even if another AOI is added later.
            aoi_seed = int.from_bytes(hashlib.sha256(f"{seed}:{aoi_id}".encode()).digest()[:8], "big")
            rng = np.random.default_rng(aoi_seed)
            order = rng.permutation(len(candidates))
            if max_per_aoi is not None:
                order = order[:max_per_aoi]

            aoi_records = 0
            for index in order:
                row_off, col_off, valid_fraction = candidates[int(index)]
                patch_id = f"{aoi_id}_r{row_off:06d}_c{col_off:06d}"
                feature_out = output_root / aoi_id / "features" / f"{patch_id}.npz"
                label_out = output_root / aoi_id / "labels" / f"{patch_id}.npz"
                if not args.overwrite and (feature_out.exists() or label_out.exists()):
                    raise FileExistsError(f"Output exists for {patch_id}; rerun with --overwrite")
                window = Window(col_off, row_off, size, size)
                feature_patch = features.read(window=window).astype(np.float32, copy=False)
                label_patch = labels.read(1, window=window).astype(np.uint8, copy=False)
                if not np.isfinite(feature_patch).all():
                    raise ValueError(f"{patch_id}: non-finite feature value")
                positive = int(np.count_nonzero(label_patch == 1))
                negative = int(np.count_nonzero(label_patch == 0))
                atomic_npz(feature_out, features=feature_patch)
                atomic_npz(label_out, labels=label_patch)
                records.append({
                    "patch_id": patch_id, "aoi_id": aoi_id, "aoi_role": role,
                    "row_off": row_off, "col_off": col_off, "patch_size": size,
                    "feature_patch_path": feature_out.relative_to(root).as_posix(),
                    "label_patch_path": label_out.relative_to(root).as_posix(),
                    "valid_fraction": f"{valid_fraction:.8f}", "positive_pixels": positive,
                    "negative_pixels": negative, "loss_fraction": f"{positive / (positive + negative):.8f}",
                    "feature_sha256": sha256(feature_out), "label_sha256": sha256(label_out),
                })
                aoi_records += 1
            aoi_summaries.append({"aoi_id": aoi_id, "aoi_role": role,
                                  "eligible_candidates": len(candidates), "patches_written": aoi_records,
                                  "selection_seed": aoi_seed})

    write_csv(manifest_path, records)
    summary = {"task": "3.8", "status": "COMPLETED", "patch_size": size, "stride": stride,
               "random_seed": seed, "min_valid_fraction": min_valid,
               "max_patches_per_aoi": max_per_aoi, "total_patches": len(records), "aois": aoi_summaries,
               "manifest_sha256": sha256(manifest_path)}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Patch manifest: {manifest_path}")
    print(f"Patches written: {len(records)}")
    print("Task 3.8 patch extraction: COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())