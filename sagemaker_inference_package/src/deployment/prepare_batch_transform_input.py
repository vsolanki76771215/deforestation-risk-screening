#!/usr/bin/env python3
"""Convert an inference patch manifest into Batch Transform JSON Lines."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
REQUIRED={'patch_id','aoi_id','feature_patch_path','row_off','col_off'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model-artifact',required=True);p.add_argument('--input-manifest',required=True);p.add_argument('--project-root',default='.');p.add_argument('--output-jsonl',required=True);a=p.parse_args();root=Path(a.project_root).resolve(); artifact=joblib.load(root/a.model_artifact); archive_key=str(artifact['archive_key']); expected=(len(artifact['channel_names']),int(artifact['expected_patch_size']),int(artifact['expected_patch_size'])); frame=pd.read_csv(root/a.input_manifest); missing=REQUIRED-set(frame); 
 if missing: raise ValueError(f'Manifest is missing columns: {sorted(missing)}')
 out=root/a.output_jsonl; out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',encoding='utf-8') as stream:
  for row in frame.itertuples(index=False):
   item=row._asdict(); path=root/str(item['feature_patch_path'])
   with np.load(path,allow_pickle=False) as archive:
    if archive_key not in archive: raise KeyError(f'{path} has no {archive_key!r} array')
    array=archive[archive_key]
   if array.shape != expected: raise ValueError(f'{path} shape {array.shape}; expected {expected}')
   if not np.isfinite(array).all(): raise ValueError(f'{path} contains non-finite feature values')
   payload={k:item[k] for k in ('patch_id','aoi_id','row_off','col_off')}; payload['features']=array.tolist(); stream.write(json.dumps(payload,separators=(',',':'))+'\n')
 print(f'Batch Transform input records: {len(frame)}; output: {out}')
if __name__=='__main__': main()
