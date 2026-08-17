import importlib.util
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

MODULE = Path(__file__).parents[1] / "src" / "acquisition" / "build_sentinel2_composites.py"
spec = importlib.util.spec_from_file_location("build_sentinel2_composites", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_resolve_selected_adds_all_asset_hrefs():
    base = {"aoi_id": "a", "target_year": "2018", "item_id": "x"}
    inventory = [{**base, **{f"{asset}_href": f"https://x/{asset}" for asset in (*mod.BANDS, "SCL")}}]
    result = mod.resolve_selected([{**base, "selection_order": "1"}], inventory)
    assert all(result[0][f"{asset}_href"].endswith(asset) for asset in (*mod.BANDS, "SCL"))


def test_resolve_selected_rejects_ambiguous_match():
    import pytest
    base = {"aoi_id": "a", "target_year": "2018", "item_id": "x"}
    with pytest.raises(ValueError, match="exactly once"):
        mod.resolve_selected([base], [base, base])


def test_write_multiband_tif_preserves_band_order(tmp_path):
    path = tmp_path / "composite.tif"
    data = np.stack([np.full((4, 5), i, dtype="float32") for i in range(1, 5)])
    mod.write_tif(path, data, from_origin(0, 4, 1, 1), "EPSG:32719", -9999.0, mod.BANDS)
    with rasterio.open(path) as src:
        assert src.count == 4
        assert src.descriptions == mod.BANDS
        assert src.nodata == -9999.0


def test_manifest_schemas_include_audit_fields():
    assert {"source_href", "local_path", "sha256", "status"} <= set(mod.ACQUIRED_FIELDS)
    assert {"valid_coverage_pct", "item_ids", "sha256", "status"} <= set(mod.COMPOSITE_FIELDS)