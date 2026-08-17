"""Build deterministic leave-one-AOI-out folds and frozen holdouts for Task 3.14."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import yaml

def norm(v): return str(v).replace("\\","/")
def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="config/modeling_dataset_task_3_14.yaml"); p.add_argument("--project-root",default="."); p.add_argument("--overwrite",action="store_true"); a=p.parse_args(); root=Path(a.project_root).resolve(); cfg=yaml.safe_load((root/a.config).read_text(encoding="utf-8")); c=cfg["columns"]; roles=cfg["roles"]; split=cfg["split"]
    inp=root/cfg["inputs"]["patch_manifest"]; dataset=root/cfg["outputs"]["dataset_manifest"]; folds=root/cfg["outputs"]["fold_manifest"]; summary=root/cfg["outputs"]["split_summary"]
    if any(x.exists() for x in (dataset,folds,summary)) and not a.overwrite: raise FileExistsError("Task 3.14 output exists; pass --overwrite")
    f=pd.read_csv(inp); required=[c[k] for k in ("patch_id","feature_patch_path","label_patch_path","aoi_id","aoi_role","row","col","patch_size","loss_fraction")]; missing=[x for x in required if x not in f]
    if missing: raise ValueError(f"Patch manifest missing columns: {missing}")
    if f[c["patch_id"]].duplicated().any(): raise ValueError("Duplicate patch IDs")
    for x in (c["feature_patch_path"],c["label_patch_path"]): f[x]=f[x].map(norm)
    ps=int(split["patch_size_pixels"]); bs=int(split["spatial_block_size_pixels"]); sizes=pd.to_numeric(f[c["patch_size"]],errors="raise").astype(int)
    if not sizes.eq(ps).all(): raise ValueError("patch_size does not match configuration")
    f["loss_binary"]=pd.to_numeric(f[c["loss_fraction"]],errors="raise").gt(float(cfg["label"]["positive_when_greater_than"])).astype("int8"); f["spatial_block_row"]=pd.to_numeric(f[c["row"]],errors="raise").astype(int)//bs; f["spatial_block_col"]=pd.to_numeric(f[c["col"]],errors="raise").astype(int)//bs
    ro=pd.to_numeric(f[c["row"]],errors="raise").astype(int)%bs; co=pd.to_numeric(f[c["col"]],errors="raise").astype(int)%bs; f["fully_inside_spatial_block"]=(ro+ps<=bs)&(co+ps<=bs)
    role=f[c["aoi_role"]].astype(str); f["dataset_partition"]=role.map({roles["development"]:"development",roles["diagnostic"]:"diagnostic_test",roles["final"]:"final_test"})
    if f["dataset_partition"].isna().any(): raise ValueError("Unknown AOI role in patch manifest")
    dropped=f[(f.dataset_partition=="development") & ~f.fully_inside_spatial_block] if split.get("drop_block_boundary_patches",True) else f.iloc[0:0]; f=f.drop(dropped.index).copy()
    dev=sorted(f.loc[f.dataset_partition.eq("development"),c["aoi_id"]].astype(str).unique()); fold_rows=[]
    for n,val_aoi in enumerate(dev,1):
        part=f[f.dataset_partition.eq("development")]
        for r in part.itertuples(index=False):
            d=r._asdict(); fold_rows.append({"fold_id":f"fold_{n:02d}","patch_id":d[c["patch_id"]],"aoi_id":d[c["aoi_id"]],"fold_split":"validation" if str(d[c["aoi_id"]])==val_aoi else "train","validation_aoi":val_aoi})
    for partition in ("diagnostic_test","final_test"):
        for r in f[f.dataset_partition.eq(partition)].itertuples(index=False):
            d=r._asdict(); fold_rows.append({"fold_id":partition,"patch_id":d[c["patch_id"]],"aoi_id":d[c["aoi_id"]],"fold_split":partition,"validation_aoi":""})
    fm=pd.DataFrame(fold_rows); f.sort_values(["dataset_partition",c["aoi_id"],c["row"],c["col"],c["patch_id"]],kind="mergesort",inplace=True); fm.sort_values(["fold_id","fold_split","aoi_id","patch_id"],kind="mergesort",inplace=True)
    dataset.parent.mkdir(parents=True,exist_ok=True); f.to_csv(dataset,index=False,lineterminator="\n"); fm.to_csv(folds,index=False,lineterminator="\n"); counts=f.dataset_partition.value_counts().sort_index().to_dict(); summary.write_text(json.dumps({"task":"3.14","status":"COMPLETED","development_aois":dev,"fold_count":len(dev),"counts_by_partition":{k:int(v) for k,v in counts.items()},"dropped_block_boundary_patch_count":len(dropped),"final_holdout_access":"frozen; evaluate once after model and threshold selection"},indent=2)+"\n",encoding="utf-8")
    print(f"Development AOIs: {len(dev)}; folds: {len(dev)}"); print(f"Dropped at block boundaries: {len(dropped)}"); print(f"Dataset: {dataset}"); print(f"Folds: {folds}"); return 0
if __name__ == "__main__": raise SystemExit(main())
