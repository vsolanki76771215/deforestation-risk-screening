import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src/validation/rank_sentinel2_scenes.py"
SPEC = importlib.util.spec_from_file_location("rank_scenes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_signed_href_handles_existing_query():
    assert MODULE.signed_href("https://x/a.tif", "sig=1") == "https://x/a.tif?sig=1"
    assert MODULE.signed_href("https://x/a.tif?a=1", "sig=1") == "https://x/a.tif?a=1&sig=1"


def test_write_rows_does_not_emit_private_mask(tmp_path):
    output = tmp_path / "rows.csv"
    row = {field: "" for field in MODULE.RANKED_FIELDS}
    row["_valid_mask"] = np.ones((2, 2), dtype=bool)
    MODULE.write_rows(output, MODULE.RANKED_FIELDS, [row])
    assert "_valid_mask" not in output.read_text()


def test_ranked_schema_contains_selection_audit_fields():
    required = {"rank", "valid_coverage_pct", "ranking_score",
                "selection_order", "incremental_valid_pixels",
                "cumulative_valid_coverage_pct"}
    assert required.issubset(MODULE.RANKED_FIELDS)