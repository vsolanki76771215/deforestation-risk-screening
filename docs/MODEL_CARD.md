# Model Card: Precision-Aware Extra Trees

## Model details

| Field | Value |
|---|---|
| Model | `ExtraTreesClassifier` |
| Final artifact | `[precision_aware_extra_trees.joblib](https://github.com/vsolanki76771215/deforestation-risk-screening/releases/download/v1.0.0/precision_aware_extra_trees.joblib)` |
| Decision threshold | 0.536 |
| Input | 11 x 32 x 32 Sentinel-2-derived feature patch |
| Patch summary statistics | Mean, standard deviation, minimum, maximum, median per channel |
| Output | Vegetation-loss risk probability and thresholded alert |
| Artifact SHA-256 | `40E198846735997BBBBFAA76FD7FB790499D9FA732579343180C6D85514C33F4` |

## Intended use

Use the model to prioritize remote-sensing analyst review of areas that may show vegetation loss between the specified baseline and comparison imagery windows. The model is appropriate for batch screening under the documented feature and preprocessing contract.

## Not intended for

- Confirming illegal mining.
- Identifying the cause of vegetation loss.
- Enforcement, penalties, or other high-impact action without independent analyst review and corroboration.
- Use with changed feature order, changed patch size, different label definition, or undocumented imagery-preprocessing changes.

## Evaluation results

| Split | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 0.8696 | 0.6025 | 0.7301 | 0.4848 | 0.5827 |
| Geographic test | 0.7472 | 0.2741 | 0.1668 | 0.3957 | 0.2347 |
| Final geographic holdout | 0.8714 | 0.6738 | 0.9126 | 0.4185 | 0.5738 |

The final geographic holdout was used only for one-time final reporting, after model and threshold selection. The lower geographic-test precision demonstrates meaningful domain shift across locations.

## Data and split protocol

- Development AOIs: la_pampa, mining_training_area, laberinto_development, las_piedras_development, and inambari_development.
- Geographic test AOI: tambopata_test_area.
- Final geographic holdout AOI: iberia_final_holdout.
- Training, threshold selection, test, and final-holdout evaluation are spatially separated.
- Patches crossing configured spatial-block boundaries are dropped to reduce leakage.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Geographic domain shift | Separate geographic test/holdout reporting; analyst review required |
| Cloud/nodata artifacts | Valid-mask contract and rejected-patch reporting |
| Label uncertainty | Hansen-derived label definition documented; scope limited to vegetation loss |
| Threshold misuse | Threshold is stored with artifact and emitted in every prediction record |
| Reproducibility drift | Hashes, input manifests, feature schema, model artifact, and run report retained |

## Monitoring

For every inference run, retain the model hash, input-manifest hash, accepted/rejected counts, alert count, threshold, imagery dates, valid coverage, and AOI summary. Investigate unexpected coverage reduction, rejection rate, probability distribution, or alert rate before using results.
