"""Streamlit demo for a completed Task 3.14 patch-inference run.

Run: streamlit run streamlit_app.py -- --inference-dir reports/inference/new_aoi_2026_08
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st


def parse_inference_directory() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inference-dir", default=os.getenv("INFERENCE_OUTPUT_DIR", "reports/demo/huepetuhe_sample"))
    args, _ = parser.parse_known_args(sys.argv[1:])
    return Path(args.inference_dir)


@st.cache_data(show_spinner=False)
def load_run(directory: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    root = Path(directory)
    files = {name: root / name for name in ("predictions.csv", "aoi_summary.csv", "run_report.json", "prediction_map.geojson")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inference output(s): " + ", ".join(missing))
    predictions = pd.read_csv(files["predictions.csv"])
    summary = pd.read_csv(files["aoi_summary.csv"])
    report = json.loads(files["run_report.json"].read_text(encoding="utf-8"))
    geojson = json.loads(files["prediction_map.geojson"].read_text(encoding="utf-8"))
    return predictions, summary, report, geojson


def risk_color(probability: float) -> list[int]:
    if probability >= 0.75:
        return [214, 39, 40, 185]
    if probability >= 0.50:
        return [255, 127, 14, 170]
    return [44, 160, 44, 130]


def selected_geojson(geojson: dict, patch_ids: set[str]) -> dict:
    features = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        if str(props.get("patch_id")) in patch_ids:
            item = json.loads(json.dumps(feature))
            item["properties"]["fill_color"] = risk_color(float(item["properties"]["probability"]))
            features.append(item)
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    st.set_page_config(page_title="Deforestation Risk Monitor", page_icon="🌳", layout="wide")
    st.title("Deforestation Risk Monitor")
    st.caption("Sentinel-2 patch-level deforestation / vegetation-loss risk. This is not confirmed illegal-mining detection.")
    run_directory = parse_inference_directory()
    try:
        predictions, summary, report, geojson = load_run(str(run_directory))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        st.error(str(error))
        st.code("streamlit run streamlit_app.py -- --inference-dir reports/inference/<run_id>")
        st.stop()

    st.sidebar.header("Filters")
    available_aois = sorted(predictions["aoi_id"].astype(str).unique())
    selected_aois = st.sidebar.multiselect("AOIs", available_aois, default=available_aois)
    min_probability = st.sidebar.slider("Minimum risk probability", 0.0, 1.0, 0.0, 0.01)
    alerts_only = st.sidebar.checkbox("Show alerts only", value=False)
    filtered = predictions[predictions.aoi_id.astype(str).isin(selected_aois)].copy()
    filtered = filtered[filtered.probability >= min_probability]
    if alerts_only:
        filtered = filtered[filtered.prediction.eq(1)]

    threshold = float(report.get("threshold", predictions["threshold"].iloc[0]))
    run_id = str(report.get("run_id", "unknown"))
    first, second, third, fourth = st.columns(4)
    first.metric("Accepted patches", f"{len(predictions):,}")
    second.metric("Alerts", f"{int(predictions.prediction.sum()):,}")
    third.metric("Selected patches", f"{len(filtered):,}")
    fourth.metric("Alert threshold", f"{threshold:.3f}")
    st.caption(f"Run: `{run_id}` · Created: {report.get('created_at_utc', 'unknown')} · Source: {run_directory}")

    st.subheader("Risk map")
    map_data = selected_geojson(geojson, set(filtered.patch_id.astype(str)))
    if not map_data["features"]:
        st.info("No patches match the current filters.")
    else:
        layer = pdk.Layer(
            "GeoJsonLayer", map_data, pickable=True, stroked=True, filled=True,
            get_fill_color="properties.fill_color", get_line_color=[40, 40, 40, 170], line_width_min_pixels=1,
        )
        deck = pdk.Deck(
            layers=[layer], map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=-12.7, longitude=-69.4, zoom=7.2),
            tooltip={"html": "<b>{patch_id}</b><br/>AOI: {aoi_id}<br/>Risk: {probability}<br/>Alert: {prediction}<br/>Threshold: {threshold}"},
        )
        st.pydeck_chart(deck, use_container_width=True)
    st.caption("Green: lower risk; orange: moderate risk; red: high risk. Patch boundaries are displayed in WGS84.")

    left, right = st.columns((1, 2))
    with left:
        st.subheader("AOI summary")
        displayed_summary = summary[summary.aoi_id.astype(str).isin(selected_aois)]
        st.dataframe(displayed_summary, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Selected patch predictions")
        columns = [column for column in ("patch_id", "aoi_id", "probability", "prediction", "threshold", "row_off", "col_off") if column in filtered]
        st.dataframe(filtered[columns].sort_values("probability", ascending=False), hide_index=True, use_container_width=True)

    st.download_button("Download filtered predictions CSV", filtered.to_csv(index=False).encode("utf-8"), "filtered_predictions.csv", "text/csv")
    st.download_button("Download filtered GeoJSON", json.dumps(map_data, indent=2).encode("utf-8"), "filtered_prediction_map.geojson", "application/geo+json")

    with st.expander("Run lineage and interpretation"):
        st.json({key: report.get(key) for key in ("task_scope", "model_artifact", "model_artifact_sha256", "input_manifest", "input_manifest_sha256", "expected_patch_shape", "accepted_patch_count", "rejected_patch_count")})
        st.write("Use this dashboard to prioritize analyst review. Alerts reflect the locked model threshold and do not establish a confirmed cause of vegetation loss.")


if __name__ == "__main__":
    main()
