# Streamlit demo

Install the lightweight dashboard dependencies:

```powershell
python -m pip install -r requirements-streamlit.txt
```

Run the demo against a completed inference output folder:

```powershell
streamlit run streamlit_app.py -- --inference-dir reports\inference\new_aoi_2026_08
```

The dashboard reads `prediction_map.geojson`, `aoi_summary.csv`,
`predictions.csv`, and `run_report.json`. It provides AOI and risk filters, a
clickable map, report table, downloads, and run-lineage details. It is designed
to present deforestation / vegetation-loss risk for analyst review—not
confirmed illegal-mining detection.
