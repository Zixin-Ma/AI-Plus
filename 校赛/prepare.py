"""Stage organizer-provided public material for a BJTU development run or Docker build."""
import hashlib
import io
from pathlib import Path
import tarfile
import zipfile
import json

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"


def stage():
    sdk = ROOT.parent / "Harness_SDK/contest-harness-sdk-1.0.4.tar.gz"
    with tarfile.open(sdk) as archive:
        for member in archive.getmembers():
            if member.isfile():
                rel = Path(*Path(member.name).parts[1:])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("invalid SDK path")
                dest = WORK / "sdk" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.extractfile(member).read())
    for key in "ABC":
        origin, = (ROOT.parent / "train_origin").glob(f"cpu-{key.lower()}-*.zip")
        with zipfile.ZipFile(origin) as archive:
            info = json.loads(archive.read("baseline-info.json"))
            for name in ["checkpoint.npz", "training.json", "public/train.jsonl", "public/validation.jsonl"]:
                data = archive.read(name)
                if name == "checkpoint.npz" and info["checkpoint_sha256"] != "sha256:" + hashlib.sha256(data).hexdigest():
                    raise ValueError("baseline digest mismatch")
                dest = WORK / key / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            if (ROOT / "app/baseline_model.py").read_bytes() != archive.read("baselines/model.py"):
                raise ValueError("baseline implementation mismatch")


if __name__ == "__main__":
    stage()
