"""Protect the single-row ABI, fallback behavior and checkpoint isolation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

import contest_submission as h
import guard
from serving import prepare

HERE=Path(__file__).resolve().parent
INPUT=HERE.parents[1]/"校赛/work"


class ServingContract(unittest.TestCase):
    def test_backbones_match_batches_at_boundaries(self):
        for key in "ABC":
            model=h.Model.load(INPUT/key/"checkpoint.npz")
            for scalar in (-12.,-10.,-1.,0.,1.,10.,12.):
                row=model.mean+scalar*model.scale
                v=model.predict(row[None,:])[0]
                np.testing.assert_allclose(model.predict_one(row),v,rtol=1e-12,atol=1e-11)

    def test_anchor_mix_offset_and_corrupt_rows(self):
        model=h.Model.load(INPUT/"B/checkpoint.npz")
        x=model.mean.copy()
        for weight in (0.,.15,.65,1.):
            for delta in (-1000.,-10.,0.,10.):
                head=guard.freeze(dict(backbone_only=np.asarray(1),life_offset=np.asarray(delta)),model,weight)
                bound=prepare("B",model,head)
                expected=h.values("B",model,head,x[None,:])[0,0]
                self.assertAlmostEqual(bound(x)["value"],expected,places=10)
        for row in (np.full_like(x,np.nan),np.full_like(x,np.inf),np.zeros(len(x)+1),[x.tolist()]):
            with self.assertRaises(ValueError):
                bound(row)

    def test_engine_ramp_matches_batch_and_saturates(self):
        levels = np.array([-2., 0., 5., 10., 12.])
        np.testing.assert_allclose(guard.shift_life(levels, 2., [0., 10.]).ravel(),
                                   [0., 0., 6., 12., 14.])
        model = h.Model.load(INPUT/"C/checkpoint.npz")
        row = model.mean.copy()
        origin = float(model.predict_one(row))
        for lower, upper in [(origin+5,origin+10),(origin-5,origin+5),(origin-10,origin-5)]:
            head = guard.freeze(dict(backbone_only=np.asarray(1),life_offset=np.asarray(3.),
                                     life_ramp=np.array([lower,upper])),model,.85)
            expected = h.values("C",model,head,row[None,:])[0,0]
            self.assertAlmostEqual(prepare("C",model,head)(row)["value"], expected, places=9)

    def test_identical_logits_choose_original_tie_break(self):
        model=h.Model.load(INPUT/"A/checkpoint.npz")
        model.params["w2"].fill(0.)
        model.params["b2"].fill(0.)
        out=prepare("A",model,{})(model.mean)
        self.assertEqual([q["label"] for q in out],model.meta["classes"])
        self.assertGreater(out[0]["score"],out[1]["score"])
        self.assertAlmostEqual(sum(q["score"] for q in out),1.,places=12)

    def test_distinct_artifacts_and_wrong_task(self):
        mod=h.Model.load(INPUT/"C/checkpoint.npz")
        r=mod.mean.copy()
        sample=SimpleNamespace(dataset_key="C",task="regression",inputs=dict(sensors=r[:21].tolist(),settings=r[21:].tolist()))
        with TemporaryDirectory(dir=HERE.parent/"work/v5") as tmp:
            folders=[]
            for i in (0,1):
                f=Path(tmp)/str(i);f.mkdir()
                mod.params["b2"]+=i
                mod.save(f/"checkpoint.npz")
                folders.append(f)
            a,b=(h.predict(sample,dir)["value"] for dir in folders)
            self.assertNotEqual(a,b)
            self.assertEqual(h.predict(sample,folders[0])["value"],a)
            with self.assertRaises(ValueError):
                h.predict(SimpleNamespace(**{**vars(sample),"task":"classification"}),folders[0])


if __name__=="__main__":
    unittest.main(verbosity=2)
