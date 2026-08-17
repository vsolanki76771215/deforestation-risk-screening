# Operations Runbook

## Local dashboard demo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

The default dashboard reads the committed 250-patch demo package at `reports/demo/huepetuhe_sample/`. It is a representative sample, not the complete Huepetuhe result.

## Full local inference

1. Add a new AOI GeoJSON under `data/aoi/`.
2. Use the configured Sentinel-2 discovery, ranking, and composite scripts.
3. Build the inference-only 11-band feature stack.
4. Extract valid patches.
5. Place the approved model artifact at `models/task_3_14/precision_aware_extra_trees.joblib`.
6. Run `src/inference/run_patch_inference.py` with the model artifact, patch manifest, reference raster, output directory, and stable run ID.
7. Validate the output directory with `src/validation/validate_inference_outputs.py`.
8. Start Streamlit with `--inference-dir reports/inference/<run_id>`.

## SageMaker Batch Transform

1. Build `precision_aware_extra_trees_model.tar.gz` with `build_sagemaker_model_package.py`.
2. Upload the model tarball and prepared JSON Lines input to S3.
3. Create a SageMaker model using a compatible scikit-learn CPU inference image.
4. Submit Batch Transform with `application/jsonlines`, `SplitType=Line`, `BatchStrategy=SingleRecord`, and `AssembleWith=Line`.
5. Download the JSON Lines output.
6. Convert it to dashboard artifacts with `postprocess_batch_transform_output.py`.
7. Validate and retain the run report.

The local 25-record handler smoke test must pass before a full cloud batch run.

## Troubleshooting

| Symptom | Check |
|---|---|
| Dashboard reports missing output | Verify all five expected output files exist in `--inference-dir` |
| No map features | Verify patch IDs in `predictions.csv` match GeoJSON properties |
| Rejected or invalid patches | Review valid-mask coverage, patch schema, and feature tensor shape |
| Batch handler failure | Re-run the 25-record smoke test and confirm the model package includes `model.joblib` and `code/inference.py` |
| Unexpected alert rate | Compare run coverage, input schema, model hash, threshold, and probability distribution with prior validated runs |

## Public deployment status

The repository contains a working local Streamlit interface and tested SageMaker Batch Transform package. Before certification submission, deploy the Streamlit interface to a public cloud URL and add a documented API or explicit Batch Transform request contract accessible to reviewers.
