"""Exercise SDK train + streaming predict in the exported filesystem with unprivileged processes."""
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1] / "校赛/work"
OUT = ROOT.parent / "work/v3"

sys.path.insert(0, str(WORK / "sdk"))
from tests.support import Host


def check(root, key):
    data, previous, model = [root / n for n in ("data", "previous", "model")]
    for d in (data, previous, model):
        d.mkdir(exist_ok=True)
    def launch(action):
        return [shutil.which("chroot"), "--userspec=65534:65534", str(root), "/usr/local/bin/python3",
                "-B", "/app/harness.py", action]
    sample = next(iter((WORK/key/"public/validation.jsonl").open()))
    task = "classification" if key == "A" else "regression"
    for name in ("train.jsonl", "validation.jsonl"):
        shutil.copyfile(WORK/key/"public"/name, data/name)
    history = []
    for i in [1, 2, 3]:
        for f in previous.iterdir():
            f.unlink()
        parent = WORK/key if i == 1 else model
        names = ["checkpoint.npz"] + [aux for aux in ("head.npz","reference.npz") if (parent/aux).is_file()]
        entries = []
        for name in names:
            shutil.copyfile(parent/name, previous/name)
            entries.append({"path":name,"sha256":hashlib.sha256((previous/name).read_bytes()).hexdigest()})
        frame = dict(resource_limits={"artifact_upload_protocol":"chunked-v1"}, protocol="contest.harness.workspace.v2", prompt="Public SDK integration test.",
                     prompt_version="replay", dataset_key=key, round_index=i, task_contract={"task":task},
                     model_name="Qwen/Qwen3.8-27B", max_output_tokens=160, artifacts=[],
                     parent_checkpoint_sha256="sha256:"+entries[0]["sha256"], parent_artifact={
                     "artifact_id":"parent","artifact":{"checkpoint_paths":names,"code_paths":[]}, "files":entries})
        env = {"PATH":"/usr/local/bin:/usr/bin:/bin", "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"}
        proc = subprocess.Popen(launch("train"), stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,text=True,env=env)
        call_ids = []
        def answer(messages, options):
            ordinal = len(call_ids)
            call_id = f"replay-{key}-{i}-{ordinal + 1}"
            call_ids.append(call_id)
            review = any("Choose one of" in message["content"] for message in messages)
            response = "candidate|public gain" if review else "refit|half|refresh the condition expert"
            if review and i == 3:
                response = "parent|hold current deployment"
            return {"request_id": call_id, "choices": [{"message": {"content": response}}]}

        agent = Host(frame, answer, history=history)
        thread = agent.serve(proc)
        code = proc.wait(timeout=180)
        thread.join(5)
        if code or agent.failure or not agent.record:
            raise RuntimeError((agent.failure, proc.stderr.read()))
        assert len(call_ids) in (1, 2)
        assert agent.record["model_call_ids"] == call_ids
        assert agent.record["history_ids"] == [r["id"] for r in history]
        config = agent.record["configuration"]
        assert config["steps"] > config["old_steps"]
        assert len(call_ids) == 1 + int(config["replanned"])
        assert len(config["static_groups"]) > 1
        assert config["deployed"] in ("parent","candidate")
        if i == 3:
            assert config["deployed"] == "parent"
        history.append({"id": f"round-{key}-{i}", **agent.record})
        candidate, = (root/"workspace").glob("candidate-*")
        for f in model.iterdir():
            f.unlink()
        chosen = candidate if config["deployed"] == "candidate" else previous
        assert agent.record["selected_artifact_id"] == ("model-test-1" if chosen == candidate else "parent")
        for p in chosen.iterdir():
            shutil.copyfile(p, model/p.name)
        shutil.rmtree(candidate)
        child = subprocess.Popen(launch("predict"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
        line = json.dumps({"task":task,"inputs":json.loads(sample)["inputs"]})
        for _ in range(2):
            child.stdin.write(line+"\n")
            child.stdin.flush()
            assert select.select([child.stdout],[],[],10)[0], "missing streamed response"
            value = json.loads(child.stdout.readline())["outputs"]
            assert (isinstance(value,list) and len(value)==6) if key=="A" else (value["value"]>=0)
        child.stdin.close()
        assert child.wait(timeout=5)==0
        print(f"{key} round {i}: SDK train, parent digest, candidate upload, "
              f"{len(call_ids)} model receipt(s), acknowledgement and streaming predict OK", flush=True)


def main():
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        root=Path(tmp)
        archive=OUT/"delivery/harness.tar"
        with archive.open("rb") as file:
            export=subprocess.Popen([str(WORK/"tools/crane"),"export","-","-"],stdin=file,stdout=subprocess.PIPE)
            extract=subprocess.run(["tar","-xf","-","-C",tmp],stdin=export.stdout)
            export.stdout.close()
            if extract.returncode or export.wait():
                raise RuntimeError("could not extract the image")
        root.chmod(0o755)
        (root/"workspace").mkdir(exist_ok=True)
        os.chown(root/"workspace",65534,65534)
        result=subprocess.check_output([shutil.which("chroot"),"--userspec=65534:65534",tmp,"/usr/local/bin/python3",
                                        "-B","/app/harness.py","describe"],text=True)
        print(result,flush=True)
        assert json.loads(result)["sdk_version"]=="1.0.4"
        for key in "ABC":
            (root/"model").mkdir(exist_ok=True)
            for p in (root/"model").iterdir():
                p.unlink()
            check(root,key)


if __name__=="__main__":
    main()
