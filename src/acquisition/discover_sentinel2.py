#!/usr/bin/env python3
"""Inventory Sentinel-2 L2A STAC items without downloading imagery."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


FIELDS = [
    "aoi_id", "aoi_role", "target_year", "window_start", "window_end",
    "item_id", "acquisition_datetime", "cloud_cover_pct", "platform",
    "processing_baseline", "proj_epsg", "collection", "source_uri",
    "bbox", "required_assets_present", "missing_assets", "B02_href",
    "B03_href", "B04_href", "B08_href", "SCL_href", "local_path",
    "query_timestamp_utc", "status",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")

    aois = config.get("aois")
    if not isinstance(aois, dict) or not aois:
        raise ValueError("Configuration must define at least one AOI under 'aois'")

    for aoi_id, aoi in aois.items():
        if not isinstance(aoi_id, str) or not aoi_id.strip():
            raise ValueError("Every AOI must have a non-empty string ID")
        if not isinstance(aoi, dict):
            raise ValueError(f"AOI '{aoi_id}' must be a mapping")

        role = aoi.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"AOI '{aoi_id}' must define a non-empty 'role'")

        bounds = aoi.get("bounds_epsg4326")
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 4
            or any(not isinstance(value, (int, float)) for value in bounds)
        ):
            raise ValueError(
                f"AOI '{aoi_id}' must define four numeric values in "
                "'bounds_epsg4326'"
            )
        min_lon, min_lat, max_lon, max_lat = bounds
        if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
            raise ValueError(
                f"AOI '{aoi_id}' has invalid 'bounds_epsg4326': {bounds}"
            )
    return config


def request_json(url: str, payload: dict[str, Any], retries: int = 3) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", "User-Agent": "deforestation-capstone/1.0"}, method="POST"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("STAC request failed")


def search_items(catalog_url: str, collection: str, bbox: list[float], start: str, end: str,
                 cloud_max: float) -> list[dict[str, Any]]:
    url = f"{catalog_url.rstrip('/')}/search"
    payload: dict[str, Any] = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": cloud_max}},
        "limit": 1000,
    }
    result = request_json(url, payload)
    items = list(result.get("features", []))
    while True:
        next_link = next((link for link in result.get("links", []) if link.get("rel") == "next"), None)
        if not next_link:
            break
        next_payload = next_link.get("body", payload)
        result = request_json(next_link.get("href", url), next_payload)
        items.extend(result.get("features", []))
    # Enforce the contract client-side as well. Some STAC deployments may
    # accept but not apply the legacy ``query`` extension consistently.
    return [
        item for item in items
        if item.get("properties", {}).get("eo:cloud_cover") is not None
        and float(item["properties"]["eo:cloud_cover"]) <= cloud_max
    ]


def item_to_row(item: dict[str, Any], aoi_id: str, aoi: dict[str, Any], year: int,
                start: str, end: str, collection: str, required_assets: list[str],
                query_timestamp: str) -> dict[str, Any]:
    props = item.get("properties", {})
    assets = item.get("assets", {})
    missing = [name for name in required_assets if name not in assets]
    row = {field: "" for field in FIELDS}
    row.update({
        "aoi_id": aoi_id,
        "aoi_role": aoi["role"],
        "target_year": year,
        "window_start": start,
        "window_end": end,
        "item_id": item.get("id", ""),
        "acquisition_datetime": props.get("datetime", ""),
        "cloud_cover_pct": props.get("eo:cloud_cover", ""),
        "platform": props.get("platform", ""),
        "processing_baseline": props.get("s2:processing_baseline", ""),
        "proj_epsg": props.get("proj:epsg", ""),
        "collection": item.get("collection", collection),
        "source_uri": next((link.get("href", "") for link in item.get("links", []) if link.get("rel") == "self"), ""),
        "bbox": json.dumps(item.get("bbox", []), separators=(",", ":")),
        "required_assets_present": not missing,
        "missing_assets": ";".join(missing),
        "query_timestamp_utc": query_timestamp,
        "status": "discovered" if not missing else "missing_required_assets",
    })
    for name in required_assets:
        row[f"{name}_href"] = assets.get(name, {}).get("href", "")
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def discover(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    s2 = config["sentinel2"]
    query_timestamp = utc_now()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    windows = [s2["baseline"], s2["comparison"]]
    for aoi_id, aoi in config["aois"].items():
        for window in windows:
            items = search_items(s2["catalog_url"], s2["collection"], aoi["bounds_epsg4326"],
                                 window["start_date"], window["end_date"], s2["cloud_cover_max_pct"])
            combo_rows = [item_to_row(item, aoi_id, aoi, window["year"], window["start_date"],
                                      window["end_date"], s2["collection"], s2["required_assets"],
                                      query_timestamp) for item in items]
            rows.extend(combo_rows)
            summaries.append({
                "aoi_id": aoi_id,
                "aoi_role": aoi["role"],
                "year": window["year"],
                "item_count": len(combo_rows),
                "complete_asset_count": sum(r["required_assets_present"] is True for r in combo_rows),
                "status": "passed" if combo_rows and all(r["required_assets_present"] is True for r in combo_rows) else "failed",
            })
    return rows, summaries, query_timestamp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/geospatial.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = args.manifest or Path(config["outputs"]["sentinel2_manifest"])
    run_manifest = args.run_manifest or Path(config["outputs"]["run_manifest"])
    started = utc_now()
    try:
        rows, summaries, query_timestamp = discover(config)
        write_csv(manifest, rows)
        expected_query_count = len(config["aois"]) * 2
        success = (
            len(summaries) == expected_query_count
            and all(s["status"] == "passed" for s in summaries)
        )
        run = {
            "task": "3.2-sentinel2-stac-discovery", "mode": "metadata_only",
            "source_version": config["specification"]["source_version"],
            "started_at_utc": started, "completed_at_utc": utc_now(),
            "query_timestamp_utc": query_timestamp, "status": "passed" if success else "failed",
            "manifest": str(manifest), "total_items": len(rows), "queries": summaries,
        }
        run_manifest.parent.mkdir(parents=True, exist_ok=True)
        run_manifest.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        for summary in summaries:
            print(f"{summary['aoi_id']:24} {summary['year']}: {summary['item_count']:3} items [{summary['status']}]")
        print(f"Manifest: {manifest} ({len(rows)} rows)")
        print(f"Task 3.2 validation: {'PASSED' if success else 'FAILED'}")
        return 0 if success else 1
    except Exception as exc:
        run_manifest.parent.mkdir(parents=True, exist_ok=True)
        run_manifest.write_text(json.dumps({"task": "3.2-sentinel2-stac-discovery", "mode": "metadata_only",
            "started_at_utc": started, "completed_at_utc": utc_now(), "status": "error",
            "error_type": type(exc).__name__, "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
