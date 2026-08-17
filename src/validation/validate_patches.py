#!/usr/bin/env python3
"""Validate Task 3.8 outputs without changing them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/patch_extraction.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--summary", default="data/manifests/patch_validation_summary.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    manifest_path = root / config["patches"]["manifest"]
    size = int(config["validation"]["expected_patch_size"])
    bands = int(config["validation"]["expected_bands"])
    held_out_aoi = config["validation"]["held_out_aoi"]
    configured_held_out_role = config["validation"].get(
        "held_out_role", "geographic_holdout"
    )

    # Task 3.7 established ``geographic_holdout`` as the canonical role for
    # tambopata_test_area.  Early Task 3.8 configuration used the obsolete
    # name ``held_out_test``; accept that configuration as an alias while
    # validating the manifest against the canonical role.
    held_out_role = (
        "geographic_holdout"
        if configured_held_out_role == "held_out_test"
        else configured_held_out_role
    )
    errors: list[str] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row_number, row in enumerate(rows, start=2):
        patch_id = row["patch_id"]
        if patch_id in seen_ids:
            errors.append(f"row {row_number}: duplicate patch_id {patch_id}")
        seen_ids.add(patch_id)
        aoi_id, role = row["aoi_id"], row["aoi_role"]
        counts[aoi_id] += 1
        if aoi_id == held_out_aoi and role != held_out_role:
            errors.append(f"{patch_id}: held-out AOI role is {role!r}, expected {held_out_role!r}")
        feature_path, label_path = root / row["feature_patch_path"], root / row["label_patch_path"]
        for path, expected_hash in ((feature_path, row["feature_sha256"]), (label_path, row["label_sha256"])):
            if not path.is_file():
                errors.append(f"{patch_id}: missing {path}")
            elif digest(path) != expected_hash:
                errors.append(f"{patch_id}: checksum mismatch for {path}")
        if not feature_path.is_file() or not label_path.is_file():
            continue
        try:
            with np.load(feature_path, allow_pickle=False) as archive:
                features = archive["features"]
            with np.load(label_path, allow_pickle=False) as archive:
                labels = archive["labels"]
            if features.shape != (bands, size, size) or features.dtype != np.float32:
                errors.append(f"{patch_id}: invalid feature shape/dtype {features.shape}/{features.dtype}")
            if labels.shape != (size, size) or labels.dtype != np.uint8:
                errors.append(f"{patch_id}: invalid label shape/dtype {labels.shape}/{labels.dtype}")
            if not np.isfinite(features).all():
                errors.append(f"{patch_id}: non-finite feature values")
            if not np.isin(labels, (0, 1)).all():
                errors.append(f"{patch_id}: labels outside {{0,1}}")
            positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
            if positive != int(row["positive_pixels"]) or negative != int(row["negative_pixels"]):
                errors.append(f"{patch_id}: manifest label counts disagree with archive")
        except Exception as exc:
            errors.append(f"{patch_id}: cannot read archive: {exc}")

    summary = {"task": "3.8", "status": "PASSED" if not errors else "FAILED",
               "manifest": manifest_path.relative_to(root).as_posix(), "total_patches": len(rows),
               "held_out_aoi": held_out_aoi, "expected_held_out_role": held_out_role,
               "patches_by_aoi": dict(sorted(counts.items())), "errors": errors}
    output = root / args.summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for aoi_id, count in sorted(counts.items()):
        print(f"{aoi_id}: {count} patches")
    print(f"Validation summary: {output}")
    print(f"Task 3.8 validation: {summary['status']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())