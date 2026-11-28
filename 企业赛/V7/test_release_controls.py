"""Regressions specific to the V2 release schedule and the submitted predictors."""
import json
from pathlib import Path
import unittest

import numpy as np

import contest_submission as harness
from industrial import fields, vector


class ReleaseControlTest(unittest.TestCase):
    def test_detect_same_size_replacement_and_changed_labels(self):
        x = np.arange(12, dtype=float).reshape(6, 2)
        y = np.arange(6)
        sig = harness.release_signature(x, y)
        self.assertEqual(sig, harness.release_signature(x.copy(), y.copy()))
        self.assertNotEqual(sig, harness.release_signature(x, y + 1))
        self.assertNotEqual(sig, harness.release_signature(x + 1, y))

    def test_parent_head_remembers_its_actual_fit_after_rejection(self):
        old_release = "12"*16
        current_release = "34"*16
        head = {"fit_release": np.frombuffer(bytes.fromhex(old_release), dtype=np.uint8)}
        self.assertFalse(harness.engine_parent_uses_release(head, current_release, True))
        self.assertTrue(harness.engine_parent_uses_release(head, old_release, False))
        self.assertTrue(harness.engine_parent_uses_release({}, current_release, True))
        self.assertFalse(harness.engine_parent_uses_release({}, current_release, False))

    def test_movement_tracks_maximum_quartile_on_changed_bearing_release(self):
        bins = np.repeat(np.arange(4), 2)
        parent = np.ones((8, 1))
        new = parent.copy()
        new[6:] += 2
        self.assertAlmostEqual(harness.release_movement("B", True, True, new, parent, bins, 8), .25)
        for key, changed, head in [("B", False, True), ("B", True, False), ("C", True, True)]:
            self.assertEqual(harness.release_movement(key, changed, head, new, parent, bins, 8), 0.)

    def test_bearing_fast_path_matches_official_schema(self):
        root = Path(__file__).resolve().parents[2] / "校赛/work/B"
        model = harness.Model.load(root / "checkpoint.npz")
        schema = fields(tuple(model.meta["paths"]))
        read = harness.extractor(schema)
        with (root / "public/validation.jsonl").open() as rows:
            for _, row in zip(range(20), rows):
                sample = json.loads(row)["inputs"]
                np.testing.assert_array_equal(read(sample), vector(sample, schema))


if __name__ == "__main__":
    unittest.main(verbosity=2)
