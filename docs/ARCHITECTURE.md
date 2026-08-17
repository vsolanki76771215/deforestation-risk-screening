# System Architecture

## Purpose

This project screens Sentinel-2 imagery for patches with elevated deforestation / vegetation-loss risk in Madre de Dios, Peru. It is not a system for confirming illegal mining or for autonomous enforcement decisions.

## Training and inference flow

```mermaid
flowchart TD
    A[AOI GeoJSON] --> B[Sentinel-2 discovery]
    B --> C[Scene ranking and acquisition]
    C --> D[2018 and 2022 seasonal composites]
    H[Hansen GFC v1.13] --> I[Forest-loss labels]
    D --> E[11-band feature stack]
    I --> F[Label alignment and validation]
    E --> G[32 x 32 feature patches]
    F --> G
    G --> J[Spatial splits and model training]
    J --> K[Extra Trees model and threshold]
    K --> L[Local or SageMaker Batch Transform inference]
    L --> M[CSV, GeoJSON, summary, run report]
    M --> N[Streamlit analyst dashboard]
```

## Component contract

| Component | Input | Output | Key control |
|---|---|---|---|
| Sentinel-2 discovery | AOI, seasonal date range, cloud limit | Candidate-scene manifest | Scene IDs and query metadata recorded |
| Compositing | Selected source bands and SCL | 2018/2022 composites and valid masks | Cloud/SCL mask and coverage threshold |
| Labeling | Hansen loss year and tree cover | Binary aligned loss raster | Tree cover >= 30%; loss years 2019-2022 |
| Features | Both composites and masks | 11-band feature GeoTIFF | EPSG:32719 alignment and nodata checks |
| Patch extraction | Feature/label rasters | 32 x 32 `.npz` patches and manifest | Invalid coverage and spatial-boundary checks |
| Model | Patch summary features | Risk probability and binary alert | Locked threshold 0.536 |
| Inference report | Accepted/rejected predictions | CSV, GeoJSON, AOI summary, JSON run report | Hashes, counts, threshold, model lineage |

## Deployment architecture

```mermaid
flowchart LR
    A[Prepared feature patches] --> B[JSON Lines batch input]
    B --> C[S3 input prefix]
    C --> D[SageMaker Batch Transform]
    E[Model artifact and inference code] --> D
    D --> F[S3 JSON Lines output]
    F --> G[Postprocessing]
    G --> H[GeoJSON and summary]
    H --> I[Streamlit dashboard]
    I --> J[Human analyst review]
```

The SageMaker workflow is asynchronous batch inference. It is not a persistent real-time prediction endpoint.

## Stored outputs

- `predictions.csv`: patch-level probability, prediction, threshold, spatial offsets, and metadata.
- `rejected_patches.csv`: rejected input records and reasons.
- `aoi_summary.csv`: AOI-level patch and alert summary.
- `prediction_map.geojson`: patch polygons for mapping.
- `run_report.json`: run lineage, hashes, threshold, accepted/rejected counts, and scope statement.

Only the small repository demo package is committed. Full rasters, patch files, batch inputs, full predictions, and model binaries remain local or in cloud storage.
