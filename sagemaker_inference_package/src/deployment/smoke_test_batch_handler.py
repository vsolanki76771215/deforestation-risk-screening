#!/usr/bin/env python3
"""Run prepared JSON Lines through the SageMaker handler locally."""
from __future__ import annotations
import argparse, importlib.util, json, shutil, tempfile
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-artifact',required=True);p.add_argument('--input-jsonl',required=True);p.add_argument('--package-dir',required=True);p.add_argument('--output-jsonl',required=True);p.add_argument('--max-records',type=int,default=25);a=p.parse_args()
 handler_path=Path(a.package_dir)/'model'/'code'/'inference.py'; spec=importlib.util.spec_from_file_location('inference',handler_path); h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 with tempfile.TemporaryDirectory() as d:
  shutil.copy2(a.model_artifact,Path(d)/'model.joblib'); model=h.model_fn(d)
  source=Path(a.input_jsonl); target=Path(a.output_jsonl); target.parent.mkdir(parents=True,exist_ok=True); count=0
  with source.open(encoding='utf-8') as inp,target.open('w',encoding='utf-8') as out:
   for line in inp:
    if not line.strip() or count>=a.max_records: continue
    payload=h.input_fn(line,'application/jsonlines'); result=h.predict_fn(payload,model); body,_=h.output_fn(result,'application/json'); out.write(body+'\n'); count+=1
 if count==0: raise RuntimeError('No input records were tested')
 print(f'Batch handler smoke test: PASSED ({count} records); output: {target}')
if __name__=='__main__': main()
