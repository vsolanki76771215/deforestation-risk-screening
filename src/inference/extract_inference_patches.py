"""Extract unlabeled 32x32 feature patches for approved model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str, root: Path, manifest: Path) -> Path:
    candidate = Path(value.replace("\\", "/"))
    for path in (candidate, root / candidate, manifest.parent / candidate):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(value)


def atomic_npz(path: Path, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".npz", dir=path.parent)
    os.close(handle)
    temp = Path(temp_name)
    try:
        np.savez_compressed(temp, features=features)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    args.output_root = (args.output_root if args.output_root.is_absolute() else root / args.output_root).resolve()
    args.output_manifest = (args.output_manifest if args.output_manifest.is_absolute() else root / args.output_manifest).resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    patch_config = config["patches"]
    size, stride = int(patch_config["patch_size"]), int(patch_config["stride"])
    min_valid = float(patch_config["min_valid_fraction"])
    if size != 32:
        raise ValueError(f"Inference contract requires 32-pixel patches; config has {size}")

    with args.feature_manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("Inference feature manifest must contain exactly one AOI row")
    row = rows[0]
    aoi_id = row["aoi_id"]
    feature_path = resolve(row["feature_stack_path"], root, args.feature_manifest)
    valid_path = resolve(row["model_valid_mask_path"], root, args.feature_manifest)
    role = config["aois"].get(aoi_id, {}).get("role", "inference_only")

    records: list[dict[str, object]] = []
    with rasterio.open(feature_path) as features, rasterio.open(valid_path) as valid:
        if features.count != 11 or valid.count != 1:
            raise ValueError("Expected an 11-band feature stack and one-band valid mask")
        if (features.width, features.height, features.crs, features.transform) != (valid.width, valid.height, valid.crs, valid.transform):
            raise ValueError("Feature stack and valid mask are not aligned")
        if features.width < size or features.height < size:
            raise ValueError("AOI is smaller than one patch")
        for row_off in range(0, features.height - size + 1, stride):
            for col_off in range(0, features.width - size + 1, stride):
                window = Window(col_off, row_off, size, size)
                valid_fraction = float((valid.read(1, window=window) != 0).mean())
                if valid_fraction < min_valid:
                    continue
                patch_id = f"{aoi_id}_r{row_off:06d}_c{col_off:06d}"
                feature_out = args.output_root / aoi_id / "features" / f"{patch_id}.npz"
                if feature_out.exists() and not args.overwrite:
                    raise FileExistsError(f"Output exists for {patch_id}; rerun with --overwrite")
                patch = features.read(window=window).astype(np.float32, copy=False)
                if not np.isfinite(patch).all():
                    raise ValueError(f"{patch_id}: non-finite feature values")
                atomic_npz(feature_out, patch)
                records.append({
                    "patch_id": patch_id, "aoi_id": aoi_id, "aoi_role": role,
                    "row_off": row_off, "col_off": col_off, "patch_size": size,
                    "feature_patch_path": str(feature_out.relative_to(root)).replace("\\", "/"),
                    "valid_fraction": f"{valid_fraction:.8f}",
                    "feature_sha256": sha256(feature_out),
                })

    if not records:
        raise ValueError("No patches meet the minimum valid-pixel threshold")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Patch manifest: {args.output_manifest}")
    print(f"Patches written: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
