"""Validate Task 3.14 AOI roles, bounds, and non-overlap; write an inventory."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--config", default="config/geospatial_task_3_14.yaml"); p.add_argument("--modeling-config", default="config/modeling_dataset_task_3_14.yaml"); p.add_argument("--project-root", default="."); a = p.parse_args()
    root = Path(a.project_root).resolve()
    geo = yaml.safe_load((root / a.config).read_text(encoding="utf-8")); model = yaml.safe_load((root / a.modeling_config).read_text(encoding="utf-8"))
    roles = model["roles"]; allowed = {roles["development"], roles["diagnostic"], roles["final"]}; rows=[]; errors=[]
    for aoi_id, spec in geo["aois"].items():
        b = list(map(float, spec["bounds_epsg4326"])); role=str(spec["role"])
        if len(b)!=4 or not (b[0] < b[2] and b[1] < b[3]): errors.append(f"{aoi_id}: invalid bounds")
        if role not in allowed: errors.append(f"{aoi_id}: unsupported role {role}")
        rows.append({"aoi_id":aoi_id,"display_name":spec.get("display_name",aoi_id),"aoi_role":role,"min_lon":b[0],"min_lat":b[1],"max_lon":b[2],"max_lat":b[3]})
    for i,x in enumerate(rows):
        for y in rows[i+1:]:
            overlap = max(x["min_lon"],y["min_lon"]) < min(x["max_lon"],y["max_lon"]) and max(x["min_lat"],y["min_lat"]) < min(x["max_lat"],y["max_lat"])
            if overlap: errors.append(f"AOIs overlap: {x['aoi_id']} and {y['aoi_id']}")
    counts=pd.Series([r["aoi_role"] for r in rows]).value_counts().to_dict(); rules=model["validation"]
    if counts.get(roles["development"],0) < int(rules["minimum_development_aois"]): errors.append("too few development AOIs")
    if counts.get(roles["diagnostic"],0) != 1: errors.append("exactly one diagnostic holdout is required")
    if counts.get(roles["final"],0) != 1: errors.append("exactly one final holdout is required")
    out=root/model["outputs"]["aoi_inventory"]; out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).sort_values("aoi_id").to_csv(out,index=False,lineterminator="\n")
    print(f"AOI inventory: {out}"); print("Task 3.14 AOI expansion: " + ("PASSED" if not errors else "FAILED"))
    if errors: print("\n".join(f"ERROR: {e}" for e in errors))
    return 0 if not errors else 1
if __name__ == "__main__": raise SystemExit(main())
