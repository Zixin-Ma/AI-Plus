"""Offline SDK-contract tests of the real enterprise training loop using the public A checkpoint.

HostRound replaces the SDK wire transport and Qwen answers. Training, inference,
selection guards, restored checkpoint measurements and Qwen reply parsing are real.
Only explicitly named host-score overrides invent an adverse host measurement.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "work/v4"
INPUT = ROOT.parent / "校赛/work"
START = INPUT / "A"
sys.path[:0] = [str(HERE), str(INPUT / "sdk")]
import contest_submission as submission
from contest_harness.round import ModelReceipt


class Parent:
    def __init__(self, root):
        self.root = root
        self.artifact_id = "parent"

    def declared_paths(self):
        return ["checkpoint.npz"]

    def path(self, name):
        if name not in self.declared_paths():
            raise AssertionError(f"undeclared parent file: {name}")
        return self.root / name


class PublicData:
    def __init__(self):
        self.rows = {}
        for split in ("train", "validation"):
            path = START / "public" / f"{split}.jsonl"
            with path.open() as stream:
                self.rows[split] = list(map(json.loads, stream))

    def training_rows(self):
        return iter(self.rows["train"])

    def validation_rows(self):
        return iter(self.rows["validation"])


def historical_entry(tag, with_plan=True):
    return {"id": "round-" + tag, "selected_artifact_id": tag,
            "configuration": {"plan": {"tool": "refit", "strength": "full"}} if with_plan else {}}


class HostRound:
    model_name = submission.MODEL
    dataset_key = "A"
    max_output_tokens = 160

    def __init__(self, data, parent, planner, reviewer, history, artifacts, host_score):
        self.parent = Parent(parent)
        self.data = data
        self.round_index = len(history) + 1
        self.records = history
        self.artifacts = [{"artifact_id": tag} for tag in artifacts]
        self.responses = [planner, reviewer]
        self.host_score = host_score
        self.events = []
        self.calls = []
        self.call_ids = []
        self.restored = []
        self.record = None

    def history(self):
        self.events.append("history")
        return self.records

    def model(self, messages, **options):
        i = len(self.calls)
        stage = ["planner", "reviewer"][i]
        self.events.append(stage)
        self.calls.append(messages)
        if not 1 <= options["max_tokens"] <= self.max_output_tokens:
            raise AssertionError("invalid model budget")
        call_id = f"model-call-{stage}"
        self.call_ids.append(call_id)
        return ModelReceipt(call_id, {"choices": [{"message": {"content": self.responses[i]}}]})

    def submit_candidate(self, directory, **description):
        self.events.append("submit")
        self.paths = description["checkpoint_paths"]
        for file in self.paths + list(description["code_paths"]):
            if not (directory / file).is_file():
                raise AssertionError(f"missing declared file {file}")
        self.saved = submission.Model.load(directory / "checkpoint.npz")
        _,head,_ = submission.artifact(directory)
        xv, yv, groups = submission.projection(self.data.validation_rows(), self.saved, "A", True)
        p = submission.values("A", self.saved, head, xv)
        self.measured = submission.measures("A", yv, p, groups)
        m = self.measured["primary"] if self.host_score is None else self.host_score
        return {"artifact_id": "uploaded", "summary": {"metrics": {"macro_f1": m}}}

    def restore_artifact(self, artifact_id, directory):
        self.events.append("restore")
        if artifact_id not in [r["artifact_id"] for r in self.artifacts] or self.restored:
            raise AssertionError("restored an unlisted or additional artifact")
        self.restored.append(artifact_id)
        directory.mkdir()
        shutil.copyfile(START / "checkpoint.npz", directory / "checkpoint.npz")
        return directory

    def finish(self, **record):
        self.events.append("finish")
        self.record = record


class LoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = PublicData()

    def setUp(self):
        # Load the same runtime modules used by the release image.
        self.temp = tempfile.TemporaryDirectory(dir=WORK)
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)
        mock_app = self.tmp / "app"
        mock_app.mkdir()
        for name in ("contest_submission.py", "policy.py", "gas.py", "industrial.py", "baseline_model.py"):
            source = HERE / name if (HERE / name).exists() else ROOT / "app" / name
            shutil.copyfile(source, mock_app / name)
        self.app = mock_app
        self.damaged = self.tmp / "parent"
        self.damaged.mkdir()
        m = submission.Model.load(START / "checkpoint.npz")
        m.params["w2"].fill(0)
        m.params["b2"].fill(0)
        m.params["b2"][0] = 5
        m.save(self.damaged / "checkpoint.npz")

    def run_round(self, planner="refit|full|Improve weak class recall.",
                  reviewer="candidate|Improved measured primary and weakest class.",
                  parent=START, history=(), artifacts=(), host_score=None):
        rnd = HostRound(self.data, parent, planner, reviewer, list(history), artifacts, host_score)
        real_step = submission.continue_mlp
        def traced_step(*args):
            rnd.events.append("train")
            return real_step(*args)
        with patch.dict(os.environ, CONTEST_WORK=str(self.tmp)), \
             patch.object(submission, "__file__", str(self.app / "contest_submission.py")), \
             patch.object(submission, "continue_mlp", traced_step):
            submission.train(rnd)
        self.assertEqual([e for e in rnd.events if e != "restore"],
                         ["history", "planner", "train", "submit", "reviewer", "finish"])
        self.assertEqual(rnd.record["model_call_ids"], ["model-call-planner", "model-call-reviewer"])
        self.assertEqual(rnd.record["history_ids"], [h["id"] for h in history])
        self.assertGreater(rnd.saved.meta["steps"], submission.Model.load(parent / "checkpoint.npz").meta["steps"])
        return rnd

    def test_refit_and_deploy_after_actual_public_gain(self):
        r = self.run_round()
        self.assertIn("head.npz", r.paths)
        self.assertEqual(r.record["selected_artifact_id"], "uploaded")
        self.assertAlmostEqual(r.record["configuration"]["scores"]["candidate"]["primary"],r.measured["primary"])

    def test_planner_can_select_half_rate_adam(self):
        r = self.run_round(planner="adam|half|Test reduced learning rate.")
        self.assertEqual(r.paths, ["checkpoint.npz","reference.npz"])
        self.assertEqual(r.saved.meta["learning_rate"], .0005)
        self.assertEqual(r.record["configuration"]["plan"]["tool"], "adam")

    def test_worse_host_score_prevents_deploy(self):
        r = self.run_round(host_score=0.)
        self.assertEqual(r.record["selected_artifact_id"], "parent")
        self.assertEqual(r.record["configuration"]["scores"]["candidate"]["primary"], 0.)

    def test_invalid_plan_and_verdict(self):
        r = self.run_round(planner="not a training plan", reviewer="unknown target")
        self.assertTrue(r.record["configuration"]["plan"]["fallback"])
        self.assertEqual(r.saved.meta["learning_rate"], .0005)
        self.assertEqual(r.record["selected_artifact_id"], "parent")

    def test_unknown_history_cannot_be_selected(self):
        r = self.run_round(reviewer="history|Roll back.", host_score=0.,
                           history=[historical_entry("invisible")])
        self.assertFalse(r.restored)
        self.assertNotIn("history", r.record["configuration"]["scores"])
        self.assertEqual(r.record["selected_artifact_id"], "parent")

    def test_old_and_legacy_artifacts_cannot_be_selected(self):
        r = self.run_round(reviewer="history|Roll back.", host_score=0.,
                           history=[historical_entry("too-old"), historical_entry("legacy", False),
                                    historical_entry("parent")], artifacts=["too-old", "legacy", "parent"])
        self.assertFalse(r.restored)
        self.assertEqual(r.record["selected_artifact_id"], "parent")

    def test_freshly_measured_visible_history_requires_explicit_selection(self):
        history = [historical_entry("older"), historical_entry("last")]
        for choice, expected in (("history", "last"), ("parent", "parent")):
            with self.subTest(choice=choice):
                r = self.run_round(parent=self.damaged, history=history, artifacts=["older", "last"],
                                   host_score=0., reviewer=f"{choice}|Compared current validation results.")
                self.assertEqual(r.events[-3:], ["restore", "reviewer", "finish"])
                self.assertEqual(r.restored, ["last"])
                cfg = r.record["configuration"]
                normal = submission.Model.load(START / "checkpoint.npz")
                x, y, groups = submission.projection(self.data.validation_rows(), normal, "A", True)
                expected_score = submission.measures("A", y, normal.predict(x), groups)
                self.assertAlmostEqual(cfg["scores"]["history"]["primary"], expected_score["primary"])
                self.assertTrue(submission.eligible("A", cfg["scores"]["parent"], cfg["scores"]["history"]))
                self.assertEqual(r.record["selected_artifact_id"], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
