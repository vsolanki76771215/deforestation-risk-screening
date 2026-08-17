"""Create a small, deterministic Streamlit demo package from a full inference run.

The output contains a representative subset of patch predictions plus matching
GeoJSON. It is suitable for GitHub; the source inference run remains unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"patch_id", "aoi_id", "probability", "prediction", "threshold"}


def representative_rows(frame: pd.DataFrame, max_patches: int) -> pd.DataFrame:
    if max_patches < 2:
        raise ValueError("--max-patches must be at least 2")
    groups = [group.sort_values("probability").reset_index(drop=True) for _, group in frame.groupby("prediction")]
    if not groups:
        raise ValueError("Predictions contain no rows")
    budget, remainder = divmod(max_patches, len(groups))
    chosen: list[pd.DataFrame] = []
    for index, group in enumerate(groups):
        count = min(len(group), budget + (1 if index < remainder else 0))
        positions = pd.Series(range(count), dtype="float64")
        positions = (positions * (len(group) - 1) / max(count - 1, 1)).round().astype(int)
        chosen.append(group.iloc[positions.unique()].copy())
    return pd.concat(chosen, ignore_index=True).sort_values("probability", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-patches", default=250, type=int)
    args = parser.parse_args()

    source = args.source_dir
    output = args.output_dir
    paths = {name: source / name for name in ("predictions.csv", "prediction_map.geojson", "run_report.json")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source inference output(s): " + ", ".join(missing))

    predictions = pd.read_csv(paths["predictions.csv"])
    missing_columns = REQUIRED_COLUMNS - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing_columns)}")
    sample = representative_rows(predictions, args.max_patches)
    selected_ids = set(sample["patch_id"].astype(str))

    geojson = json.loads(paths["prediction_map.geojson"].read_text(encoding="utf-8"))
    features = [feature for feature in geojson.get("features", []) if str(feature.get("properties", {}).get("patch_id")) in selected_ids]
    missing_geometry = selected_ids - {str(feature["properties"]["patch_id"]) for feature in features}
    if missing_geometry:
        raise ValueError(f"GeoJSON has no geometry for {len(missing_geometry)} sampled patch(es)")

    source_report = json.loads(paths["run_report.json"].read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(columns=["patch_id", "aoi_id", "quality_flag", "rejection_reason"]).to_csv(
        output / "rejected_patches.csv", index=False
    )
    summary = (
        sample.groupby("aoi_id")
        .agg(
            patch_count=("patch_id", "size"),
            alert_count=("prediction", "sum"),
            mean_probability=("probability", "mean"),
            max_probability=("probability", "max"),
        )
        .reset_index()
    )
    summary["alert_rate"] = summary["alert_count"] / summary["patch_count"]
    summary.to_csv(output / "aoi_summary.csv", index=False)
    (output / "prediction_map.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )
    report = dict(source_report)
    report.update(
        {
            "run_id": f"{source_report.get('run_id', 'inference')}-demo-sample",
            "demo_sample": True,
            "source_run_id": source_report.get("run_id"),
            "source_accepted_patch_count": source_report.get("accepted_patch_count"),
            "accepted_patch_count": int(len(sample)),
            "rejected_patch_count": 0,
            "alert_count": int(sample["prediction"].sum()),
            "sampling_note": "Deterministic representative subset for the repository dashboard demo; not the full inference result.",
        }
    )
    (output / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Demo package: {output} ({len(sample)} patches; {len(features)} map features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
