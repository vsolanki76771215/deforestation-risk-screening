"""Validate Task 3.14 role isolation, geographic CV, files, and audit gates."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="config/modeling_dataset_task_3_14.yaml"); p.add_argument("--project-root",default="."); a=p.parse_args(); root=Path(a.project_root).resolve(); cfg=yaml.safe_load((root/a.config).read_text(encoding="utf-8")); c=cfg["columns"]; roles=cfg["roles"]; checks=[]; errors=[]
    def check(name,ok,detail): checks.append({"name":name,"passed":bool(ok),"detail":detail}); errors.extend([] if ok else [f"{name}: {detail}"])
    data=pd.read_csv(root/cfg["outputs"]["dataset_manifest"]); folds=pd.read_csv(root/cfg["outputs"]["fold_manifest"]); audit=pd.read_csv(root/cfg["outputs"]["label_audit"]); inventory=pd.read_csv(root/cfg["outputs"]["aoi_inventory"])
    check("unique_patch_ids",data[c["patch_id"]].is_unique,f"duplicates={data[c['patch_id']].duplicated().sum()}")
    dev=set(data.loc[data.dataset_partition.eq("development"),c["aoi_id"]].astype(str)); original=set(cfg["validation"]["original_development_aois"]); check("development_aoi_count",len(dev)>=int(cfg["validation"]["minimum_development_aois"]),f"count={len(dev)}"); check("new_development_aoi_count",len(dev-original)>=int(cfg["validation"]["minimum_new_development_aois"]),f"new={sorted(dev-original)}")
    final_ids=set(data.loc[data.dataset_partition.eq("final_test"),c["patch_id"]].astype(str)); diag_ids=set(data.loc[data.dataset_partition.eq("diagnostic_test"),c["patch_id"]].astype(str)); development_folds=folds[folds.fold_id.str.startswith("fold_")]
    check("final_absent_from_cv",final_ids.isdisjoint(set(development_folds.patch_id.astype(str))),"final holdout cannot appear in CV"); check("diagnostic_absent_from_cv",diag_ids.isdisjoint(set(development_folds.patch_id.astype(str))),"diagnostic holdout cannot appear in CV")
    validation_aois=set()
    for fold_id,g in development_folds.groupby("fold_id"):
        va=set(g.loc[g.fold_split.eq("validation"),"aoi_id"].astype(str)); tr=set(g.loc[g.fold_split.eq("train"),"aoi_id"].astype(str)); check(f"{fold_id}_one_validation_aoi",len(va)==1,f"validation={sorted(va)}"); check(f"{fold_id}_aoi_isolation",va.isdisjoint(tr),f"overlap={sorted(va&tr)}"); validation_aois |= va
    check("every_development_aoi_validated",validation_aois==dev,f"expected={sorted(dev)}, actual={sorted(validation_aois)}")
    blocks=data[data.dataset_partition.eq("development")].groupby([c["aoi_id"],"spatial_block_row","spatial_block_col"]).size(); check("development_blocks_exist",len(blocks)>0,f"blocks={len(blocks)}"); check("label_audit_passed",audit.status.astype(str).eq("PASSED").all(),f"failed={audit.loc[audit.status.ne('PASSED'),'aoi_id'].tolist()}")
    check("inventory_roles",set(inventory.aoi_role)=={roles["development"],roles["diagnostic"],roles["final"]},f"roles={sorted(set(inventory.aoi_role))}")
    if cfg["validation"].get("require_both_classes_per_evaluation_aoi",True):
        bad=data.groupby(c["aoi_id"])["loss_binary"].nunique(); check("both_classes_per_aoi",bad.ge(2).all(),f"bad={bad[bad<2].to_dict()}")
    if cfg["validation"].get("require_existing_patch_files",True):
        paths=[(Path(v) if Path(v).is_absolute() else root/v) for col in (c["feature_patch_path"],c["label_patch_path"]) for v in data[col].astype(str)]; missing=[str(x) for x in paths if not x.is_file()]; check("patch_files_exist",not missing,f"missing={len(missing)}, examples={missing[:5]}")
        if not missing and cfg["validation"].get("check_npz_readable",True):
            unread=[]
            for x in paths:
                try:
                    with np.load(x,allow_pickle=False) as z:
                        if not z.files: raise ValueError("empty archive")
                except Exception as e: unread.append(f"{x}: {e}")
            check("npz_readable",not unread,f"unreadable={len(unread)}, examples={unread[:3]}")
    result={"task":"3.14","status":"PASSED" if not errors else "FAILED","row_count":len(data),"development_aoi_count":len(dev),"fold_count":development_folds.fold_id.nunique(),"checks":checks,"errors":errors}; out=root/cfg["outputs"]["validation_summary"]; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(f"Validation summary: {out}"); print(f"Task 3.14 validation: {result['status']}"); return 0 if not errors else 1
if __name__ == "__main__": raise SystemExit(main())
