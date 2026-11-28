"""Build a Linux/amd64 Docker-save archive on a server without a Docker daemon."""
import argparse
import importlib.util
import py_compile
import sys
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent / "work/v7"
INPUT = ROOT.parents[1] / "校赛/work"

BASE_DIGEST = "sha256:1475c4f5b6bcb7d35e57661d6f5d7429ec4041275633c15f2b49350ac898276d"
WHEEL_SHA = "fd83c01228a688733f1ded5201c678f0c53ecc1006ffbc404db9f7a899ac6249"


def add(archive, name, data, mode=0o644):
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), mode
    archive.addfile(info, io.BytesIO(data))


def add_runtime(archive, name, data, mode, temp, cache):
    add(archive, name, data, mode)
    if cache and name.endswith(".py"):
        source, compiled = temp/"source.py", temp/"source.pyc"
        source.write_bytes(data)
        py_compile.compile(str(source), cfile=str(compiled), dfile="/"+name, doraise=True,
                           invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
        add(archive, importlib.util.cache_from_source(name), compiled.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bytecode", action="store_true", help="build the matched control image")
    args = parser.parse_args()
    if sys.implementation.cache_tag != "cpython-312":
        raise RuntimeError("Use the project CPython 3.12 runtime to build the image")
    deps = INPUT / "tools"
    base = deps / "base.tar"
    wheel, = deps.glob("numpy-2.2.6-cp312-*.whl")
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA:
        raise ValueError("unexpected NumPy wheel")
    out = WORK / ("image-controls/plain" if args.no_bytecode else "delivery")
    files = [ROOT/(name+".py") for name in ("baseline_model","industrial","contest_submission",
             "policy","guard","serving","prototype","gas","bearing","engine")]
    out.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        def emit(archive,name,data,mode=0o644):
            add_runtime(archive,name,data,mode,Path(tmp),not args.no_bytecode)
        layer = Path(tmp) / "app.tar"
        with tarfile.open(layer, "w") as archive, zipfile.ZipFile(wheel) as pkg:
            for item in pkg.infolist():
                if item.is_dir() or ".data/" in item.filename or "tests" in Path(item.filename).parts:
                    continue
                name = "usr/local/lib/python3.12/site-packages/" + item.filename
                emit(archive, name, pkg.read(item), ((item.external_attr >> 16) & 0o777) or 0o644)
            for file in sorted((INPUT / "sdk/contest_harness").glob("*.py")):
                emit(archive, "app/contest_harness/" + file.name, file.read_bytes())
            emit(archive, "app/harness.py", (INPUT / "sdk/template/harness.py").read_bytes())
            for file in files:
                emit(archive, "app/" + file.name, file.read_bytes())
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
            add(dst, "manifest.json", json.dumps([{"Config":cname,"RepoTags":["bjtu-rsi:enterprise-v7"],"Layers":layers}]).encode())
        tmp_image.replace(out / "harness.tar")
    print(out / "harness.tar")


if __name__ == "__main__":
    main()
