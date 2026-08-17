# Data Sheet

## Data sources

| Source | Use | Version / contract |
|---|---|---|
| Sentinel-2 Level-2A | Optical imagery for features | STAC collection `sentinel-2-l2a`; B02, B03, B04, B08, SCL |
| Hansen Global Forest Change | Forest-loss training labels | GFC-2025-v1.13; tree cover 2000 and loss year |
| Project AOI GeoJSON | Study/training/inference boundaries | WGS84 polygons stored in `data/aoi/` |

## Collection and preprocessing

- Study region: Madre de Dios, Peru.
- Baseline imagery window: June-September 2018.
- Comparison imagery window: June-September 2022.
- Candidate-scene cloud cover limit: 20%.
- Analysis CRS: EPSG:32719.
- Invalid SCL classes are excluded; both baseline and comparison pixels must be valid.
- Selected scenes, acquisition details, coverage, and composite inputs are tracked in versioned manifests.

## Label definition

- Eligible forest: Hansen tree cover 2000 >= 30%.
- Positive class: loss year 2019-2022.
- Earlier loss is excluded from the positive target.
- The label expresses observed forest-loss timing in the Hansen product; it does not identify a cause.

## Feature contract

The feature stack has 11 channels: four 2018 reflectance bands, four 2022 reflectance bands, NDVI for each year, and 2022-minus-2018 NDVI change. The model consumes 32 x 32 valid feature patches and summarizes each channel with fixed statistics.

## Repository data policy

The repository includes AOI GeoJSON files, manifests, selected metric reports, and a 250-patch dashboard demonstration package. It excludes raw imagery, source rasters, processed GeoTIFFs, `.npz` patch files, full patch manifests, full prediction output, and model binaries because they are large derived artifacts.

To reproduce full data processing, use the scripts and configuration files with original Sentinel-2 and Hansen sources. Keep all regenerated artifacts outside normal Git history and record their hashes/manifests.

## Privacy, license, and ethical use

This project uses environmental geospatial data and contains no personal data. Use Sentinel-2/Copernicus and Hansen Global Forest Change data according to their applicable attribution and terms. Do not represent model alerts as confirmed illegal activity.
