"""Regression tests for the recurring-context policy and exact fallback."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import benchmark
import contest_submission as h
import guard

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "校赛/work"


class PolicyTest(unittest.TestCase):
    def test_context_is_bounded_and_retrieves_only_recent_matching_releases(self):
        model = h.Model.load(DATA / "C/checkpoint.npz")
        x = np.vstack((model.mean + model.scale, model.mean))
        profile = h.context_profile(model, x)
        def record(i, vector):
            return {"id": i, "configuration": {"context_profile": vector.tolist()}}
        records = [record(i, profile + 4.) for i in range(8)]
        records[0] = record(0, profile)
        records[4] = record(4, profile + .1)
        records[5] = record(5, profile)
        records[6] = {"id": 6, "configuration": {}}
        names = [r["id"] for _, _, r, _ in h.prior_contexts(records, profile, .18)]
        self.assertEqual(names, [5,4])
        self.assertEqual(h.prior_contexts(records, profile+2., .18), [])

    def test_exact_static_prediction_survives_checkpoint_restore(self):
        model = h.Model.load(DATA / "B/checkpoint.npz")
        with (DATA/"B/public/validation.jsonl").open() as f:
            samples = [json.loads(next(f)) for _ in range(10)]
        x,_,_ = h.projection(samples, model, "B")
        expected = model.predict(x)
        with TemporaryDirectory(dir=HERE.parent/"work/v6") as directory:
            out = Path(directory)
            anchor = {k:v.copy() for k,v in model.params.items()}
            model.params["b2"] += 1
            model.save(out/"checkpoint.npz")
            np.savez_compressed(out/"reference.npz", **anchor)
            np.savez_compressed(out/"head.npz", backbone_only=1, expert_weight=0.)
            restored, head, _ = h.artifact(out)
            np.testing.assert_allclose(h.values("B", restored, head, x)[:,0], expected, atol=1e-11)
            self.assertGreater(np.max(np.abs(expected-restored.predict(x))), 0.)
            with self.assertRaises(ValueError):
                h.values("B", restored, head, np.full_like(x, np.nan))

    def test_rotating_blocks_exclude_audit_rows_and_whole_engines(self):
        for key in "ABC":
            with (DATA/key/"public/validation.jsonl").open() as stream:
                rows = [json.loads(line) for line in stream]
            for split in [8,9,10,11,12,13]:
                _, phases, audit = benchmark.partitions(rows,key,split)
                self.assertTrue(all(np.any((phases==p) & audit) and np.any((phases==p) & ~audit) for p in range(4)))
                if key == "C":
                    a,b = set(),set()
                    for row,held in zip(rows,audit):
                        meta=row["meta"]
                        (a if held else b).add((meta["source_subset"],meta["unit"]))
                    self.assertFalse(a & b)

    def test_paired_f1_influence_is_centered(self):
        y = np.array([0,0,1,1,2,2])
        p = np.eye(3)[[0,1,1,2,2,0]]
        infl = h.f1_influence(y, p)
        self.assertAlmostEqual(float(infl.mean()), 0., places=12)
        np.testing.assert_array_equal(infl-h.f1_influence(y,p.copy()), np.zeros(len(y)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
