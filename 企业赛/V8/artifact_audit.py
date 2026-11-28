"""Check the exported Docker archive against frozen source, original wheel and SDK."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import qualify

ROOT = Path(__file__).resolve().parent
WORK = qualify.OUT
OFFICIAL = ROOT.parents[1] / "校赛/work"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    lock = json.loads((WORK / "freeze.json").read_text())
    assert qualify.manifest() == lock
    image = WORK / "delivery/harness.tar"
    with tarfile.open(image) as archive:
        record, = json.load(archive.extractfile("manifest.json"))
        config_bytes = archive.extractfile(record["Config"]).read()
        assert sha(config_bytes) + ".json" == record["Config"]
        cfg = json.loads(config_bytes)
        digests = []
        for index, path in enumerate(record["Layers"]):
            data = archive.extractfile(path).read()
            digest = "sha256:" + sha(data)
            assert digest == cfg["rootfs"]["diff_ids"][index]
            digests.append(digest)
        with tarfile.open(fileobj=io.BytesIO(data)) as app:
            names = set(app.getnames())
            code = {name: sha(app.extractfile("app/" + name).read()) for name in lock["code"] if name != "benchmark.py"}
            assert all(code[name] == lock["code"][name] for name in code)
            sdk = {path.name: sha(app.extractfile("app/contest_harness/" + path.name).read())
                   for path in (OFFICIAL / "sdk/contest_harness").glob("*.py")}
            assert all(sdk[name] == qualify.sha(OFFICIAL / "sdk/contest_harness" / name) for name in sdk)
            assert sha(app.extractfile("app/harness.py").read()) == qualify.sha(OFFICIAL / "sdk/template/harness.py")
            assert all(not n.endswith((".jsonl", ".npz", ".pem", ".env")) for n in names if n.startswith("app/"))
            wheel, = (OFFICIAL / "tools").glob("numpy-2.2.6-cp312-*.whl")
            with zipfile.ZipFile(wheel) as dist:
                notices = [n for n in dist.namelist() if "dist-info" in n and "LICENSE" in n]
                assert notices
                for n in notices:
                    assert app.extractfile("usr/local/lib/python3.12/site-packages/" + n).read() == dist.read(n)
    assert cfg["config"]["User"] == "65534:65534" and record["RepoTags"] == ["bjtu-rsi:enterprise-v8"]
    output = dict(archive_sha256=qualify.sha(image), archive_bytes=image.stat().st_size, docker_config=record["Config"],
                  layers=digests, deployment_modules=code, sdk_modules=sdk, wheel_notices=notices,
                  user=cfg["config"]["User"], entrypoint=cfg["config"]["Entrypoint"], tag=record["RepoTags"][0])
    (WORK / "artifact-audit.json").write_text(json.dumps(output, indent=2))
    print(output["archive_sha256"], len(code), len(sdk))


if __name__ == "__main__":
    main()
