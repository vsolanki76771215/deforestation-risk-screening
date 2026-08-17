"""Audit temporal alignment, coverage, and class prevalence for every Task 3.14 AOI."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

def first(row, names, default=None):
    for n in names:
        if n in row and pd.notna(row[n]): return row[n]
    return default

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="config/modeling_dataset_task_3_14.yaml"); p.add_argument("--project-root",default="."); a=p.parse_args(); root=Path(a.project_root).resolve()
    cfg=yaml.safe_load((root/a.config).read_text(encoding="utf-8")); geo=yaml.safe_load((root/cfg["inputs"]["geospatial_config"]).read_text(encoding="utf-8")); s2=pd.read_csv(root/cfg["inputs"]["sentinel2_composite_manifest"]); labels=pd.read_csv(root/cfg["inputs"]["label_manifest"]); rules=cfg["validation"]; rows=[]
    for aoi,spec in geo["aois"].items():
        sr=s2[s2.get("aoi_id",s2.get("aoi",pd.Series(dtype=str))).astype(str).eq(aoi)]; lr=labels[labels.get("aoi_id",labels.get("aoi",pd.Series(dtype=str))).astype(str).eq(aoi)]
        years=set(pd.to_numeric(sr.get("target_year",pd.Series(dtype=float)),errors="coerce").dropna().astype(int)); cover=pd.to_numeric(sr.get("valid_coverage_pct",pd.Series(dtype=float)),errors="coerce")
        label=lr.iloc[0].to_dict() if len(lr)==1 else {}; eligible=float(first(label,["eligible_pixels"],0) or 0); pos=float(first(label,["positive_pixels"],0) or 0); neg=float(first(label,["negative_pixels"],0) or 0)
        eligible_cov=float(first(label,["eligible_label_coverage_pct","eligible_coverage_pct"],100.0 if eligible>0 else 0.0)); prevalence=100.0*pos/(pos+neg) if pos+neg else 0.0
        checks={"has_two_composites":{2018,2022}.issubset(years),"composite_coverage":len(cover)>=2 and cover.ge(float(rules["minimum_valid_coverage_pct"])).all(),"single_label_row":len(lr)==1,"eligible_coverage":eligible_cov>=float(rules["require_eligible_label_coverage_pct"]),"both_classes":pos>0 and neg>0}
        rows.append({"aoi_id":aoi,"aoi_role":spec["role"],"baseline_year":2018,"comparison_year":2022,"min_valid_coverage_pct":float(cover.min()) if len(cover) else 0.0,"eligible_label_coverage_pct":eligible_cov,"positive_pixels":int(pos),"negative_pixels":int(neg),"positive_pct":prevalence,"status":"PASSED" if all(checks.values()) else "FAILED","failed_checks":"|".join(k for k,v in checks.items() if not v)})
    out=root/cfg["outputs"]["label_audit"]; out.parent.mkdir(parents=True,exist_ok=True); result=pd.DataFrame(rows).sort_values("aoi_id"); result.to_csv(out,index=False,lineterminator="\n"); print(result[["aoi_id","status","positive_pct"]].to_string(index=False)); print(f"Label audit: {out}"); return 0 if result.status.eq("PASSED").all() else 1
if __name__ == "__main__": raise SystemExit(main())
