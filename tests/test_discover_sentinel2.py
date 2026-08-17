import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "acquisition" / "discover_sentinel2.py"
SPEC = importlib.util.spec_from_file_location("discover_sentinel2", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiscoveryTests(unittest.TestCase):
    def test_config_has_frozen_aois(self):
        config = MODULE.load_config(Path(__file__).parents[1] / "config" / "geospatial.yaml")
        self.assertEqual(config["aois"]["tambopata_test_area"]["role"], "held_out_test")

    def test_item_mapping_detects_missing_asset(self):
        item = {"id": "x", "collection": "sentinel-2-l2a", "properties": {"datetime": "2022-07-01T00:00:00Z", "eo:cloud_cover": 4},
                "assets": {name: {"href": name} for name in ["B02", "B03", "B04", "B08"]}, "links": [], "bbox": [1, 2, 3, 4]}
        row = MODULE.item_to_row(item, "a", {"role": "train"}, 2022, "2022-06-01", "2022-09-30",
                                 "sentinel-2-l2a", ["B02", "B03", "B04", "B08", "SCL"], "now")
        self.assertEqual(row["status"], "missing_required_assets")
        self.assertEqual(row["missing_assets"], "SCL")

    def test_csv_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.csv"
            MODULE.write_csv(path, [])
            self.assertEqual(path.read_text().splitlines()[0].split(","), MODULE.FIELDS)

    def test_search_enforces_cloud_limit_client_side(self):
        original = MODULE.request_json
        MODULE.request_json = lambda *_args, **_kwargs: {
            "features": [
                {"id": "keep", "properties": {"eo:cloud_cover": 20}},
                {"id": "drop", "properties": {"eo:cloud_cover": 20.1}},
                {"id": "unknown", "properties": {}},
            ],
            "links": [],
        }
        try:
            items = MODULE.search_items("https://example.test", "collection", [0, 0, 1, 1],
                                        "2022-06-01", "2022-09-30", 20)
        finally:
            MODULE.request_json = original
        self.assertEqual([item["id"] for item in items], ["keep"])


if __name__ == "__main__":
    unittest.main()