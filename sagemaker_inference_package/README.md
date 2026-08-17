# SageMaker Batch Transform inference package

This package deploys the approved Task 3.14 model as a CPU-based SageMaker
Batch Transform job.  It retains the local inference contract exactly:

- input patch shape: `11 x 32 x 32`
- per-channel statistics: `mean`, `std`, `min`, `max`, `median`
- model threshold: read from the approved model artifact (`0.536` for the
  current precision-aware model)
- result: vegetation-loss risk probability, not proof of illegal mining

## Package the model

From the `deforestation-capstone` project root in PowerShell:

```powershell
python sagemaker_inference_package\src\deployment\build_sagemaker_model_package.py `
  --model-artifact models\task_3_14\precision_aware_extra_trees.joblib `
  --package-dir sagemaker_inference_package `
  --output-model-tar-gz artifacts\sagemaker\precision_aware_extra_trees_model.tar.gz
```

Upload the resulting `.tar.gz` to S3.  It is the `ModelDataUrl` for the
SageMaker model.  Create a SageMaker model that references this S3 model artifact and a compatible SageMaker scikit-learn CPU inference image, then submit a Batch Transform job. This package implements an asynchronous batch-inference contract; it does not create a persistent real-time endpoint.

## Prepare Batch Transform input

```powershell
python sagemaker_inference_package\src\deployment\prepare_batch_transform_input.py `
  --model-artifact models\task_3_14\precision_aware_extra_trees.joblib `
  --input-manifest data\manifests\inference_patches_huepetuhe.csv `
  --project-root . `
  --output-jsonl artifacts\sagemaker\huepetuhe_input.jsonl
```

Each JSON line contains its original patch metadata plus the `features` tensor.
Create a 25-record input file for the cloud smoke test:

```powershell
Get-Content artifacts\sagemaker\huepetuhe_input.jsonl -TotalCount 25 |
  Set-Content artifacts\sagemaker\huepetuhe_smoke_input.jsonl
```

For the full run, shard the input before upload so multiple Batch Transform
instances can work in parallel:

```powershell
python sagemaker_inference_package\src\deployment\shard_batch_transform_input.py `
  --input-jsonl artifacts\sagemaker\huepetuhe_input.jsonl `
  --output-dir artifacts\sagemaker\huepetuhe_input_shards `
  --records-per-shard 500
```

Set Batch Transform to `application/jsonlines`, `SplitType=Line`,
`BatchStrategy=SingleRecord`, `AssembleWith=Line`, and use one instance for the
cloud smoke test.  `SingleRecord` is required because the handler validates one
feature-patch JSON object at a time.

## Local smoke test

This validates the same container handler SageMaker will call, using the real
model and a small batch of prepared records:

```powershell
python sagemaker_inference_package\src\deployment\smoke_test_batch_handler.py `
  --model-artifact models\task_3_14\precision_aware_extra_trees.joblib `
  --input-jsonl artifacts\sagemaker\huepetuhe_smoke_input.jsonl `
  --package-dir sagemaker_inference_package `
  --output-jsonl artifacts\sagemaker\huepetuhe_smoke_output.jsonl `
  --max-records 25
```

Expected result: `Batch handler smoke test: PASSED` and every output includes
the incoming patch metadata, probability, prediction, threshold, and a
quality flag.

## Convert SageMaker output for Streamlit

After Batch Transform completes, download its `.out` JSON-lines output and run:

```powershell
python sagemaker_inference_package\src\deployment\postprocess_batch_transform_output.py `
  --batch-output-jsonl artifacts\sagemaker\huepetuhe_full_output.jsonl `
  --reference-raster data\processed\inference_features_huepetuhe\huepetuhe_inference_area\model_features.tif `
  --output-dir reports\inference\huepetuhe_sagemaker_2018_2022 `
  --run-id huepetuhe-sagemaker-2026-08
```

This writes the dashboard inputs: `predictions.csv`, `rejected_patches.csv`,
`aoi_summary.csv`, `prediction_map.geojson`, and `run_report.json`.
