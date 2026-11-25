"""Build a Linux/amd64 Docker-save archive on a server without a Docker daemon."""
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
import zipfile
from pathlib import Path

from prepare import ROOT, WORK

BASE_DIGEST = "sha256:1475c4f5b6bcb7d35e57661d6f5d7429ec4041275633c15f2b49350ac898276d"
WHEEL_SHA = "fd83c01228a688733f1ded5201c678f0c53ecc1006ffbc404db9f7a899ac6249"


def add(archive, name, data, mode=0o644):
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), mode
    archive.addfile(info, io.BytesIO(data))


def main():
    deps = WORK / "tools"
    base = deps / "base.tar"
    wheel, = deps.glob("numpy-2.2.6-cp312-*.whl")
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA:
        raise ValueError("unexpected NumPy wheel")
    out = WORK / "delivery"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        layer = Path(tmp) / "app.tar"
        with tarfile.open(layer, "w") as archive, zipfile.ZipFile(wheel) as pkg:
            for item in pkg.infolist():
                if item.is_dir() or ".data/" in item.filename:
                    continue
                name = "usr/local/lib/python3.12/site-packages/" + item.filename
                add(archive, name, pkg.read(item), ((item.external_attr >> 16) & 0o777) or 0o644)
            for file in sorted((WORK / "sdk/contest_harness").glob("*.py")):
                add(archive, "app/contest_harness/" + file.name, file.read_bytes())
            add(archive, "app/harness.py", (WORK / "sdk/template/harness.py").read_bytes())
            for file in sorted((ROOT / "app").glob("*.py")):
                add(archive, "app/" + file.name, file.read_bytes())
        tmp_image = Path(tmp) / "image.tar"
        with tarfile.open(base) as src, tarfile.open(tmp_image, "w") as dst:
            manifest, = json.load(src.extractfile("manifest.json"))
            raw_config = src.extractfile(manifest["Config"]).read()
            if hashlib.sha256(raw_config).hexdigest() != "f4a31709c79ea38eafccf1234211e87d36625b5f9752d3df28319b43b48c0f23":
                raise ValueError("unexpected base config")
            config = json.loads(raw_config)
            layers = []
            for srcname, diff in zip(manifest["Layers"],config["rootfs"]["diff_ids"],strict=True):
                packed = src.extractfile(srcname).read()
                data = gzip.decompress(packed) if packed[:2] == b"\x1f\x8b" else packed
                digest = hashlib.sha256(data).hexdigest()
                if "sha256:" + digest != diff:
                    raise ValueError("base layer digest mismatch")
                layers.append(digest + "/layer.tar")
                add(dst, layers[-1], data)
            data = layer.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            layers.append(digest + "/layer.tar")
            add(dst, layers[-1], data)
            config["rootfs"]["diff_ids"].append("sha256:" + digest)
            config.setdefault("history", []).append({"created_by":"Install SDK 1.0.4, NumPy and participant harness"})
            cfg = config["config"]
            cfg.update(User="65534:65534", WorkingDir="/app", Entrypoint=["python3","/app/harness.py"], Cmd=["describe"])
            cfg["Env"] += ["PYTHONDONTWRITEBYTECODE=1", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1"]
            raw = json.dumps(config, separators=(",", ":")).encode()
            cname = hashlib.sha256(raw).hexdigest() + ".json"
            add(dst, cname, raw)
            add(dst, "manifest.json", json.dumps([{"Config":cname,"RepoTags":["bjtu-rsi:campus-v1"],"Layers":layers}]).encode())
        tmp_image.replace(out / "harness.tar")
    print(out / "harness.tar")


if __name__ == "__main__":
    main()
