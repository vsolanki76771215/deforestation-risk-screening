#!/usr/bin/env python3
"""Build a SageMaker-compatible model.tar.gz without changing the model."""
from __future__ import annotations
import argparse, shutil, tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-artifact',required=True); p.add_argument('--package-dir',required=True); p.add_argument('--output-model-tar-gz',required=True); a=p.parse_args()
    source=Path(a.model_artifact).resolve(); package=Path(a.package_dir).resolve(); output=Path(a.output_model_tar_gz).resolve()
    code=package/'model'/'code'
    if not source.is_file(): raise FileNotFoundError(source)
    if not (code/'inference.py').is_file(): raise FileNotFoundError(code/'inference.py')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.parent/'_sagemaker_model_staging'
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(); shutil.copy2(source, staging/'model.joblib'); shutil.copytree(code, staging/'code')
    with tarfile.open(output,'w:gz') as archive:
        for path in sorted(staging.rglob('*')): archive.add(path,arcname=path.relative_to(staging))
    shutil.rmtree(staging)
    print(f'SageMaker model artifact: {output}')
if __name__=='__main__': main()
