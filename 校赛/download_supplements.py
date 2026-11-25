"""Download manifest-listed public archives with resumable ranges and integrity receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import urlparse
import zipfile

import requests

LOCAL = threading.local()
HOST_LOCKS = {}
PRINT_LOCK = threading.Lock()

def announce(*items):
    with PRINT_LOCK:
        print(*items, flush=True)

def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)

def digest(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()

def receive_piece(job, index, start, end, parts, proxy):
    target = parts / f"{index:06}.part"
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        return expected
    temporary = target.with_suffix(".receiving")
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers["User-Agent"] = "AIPlus-Public-Dataset-Downloader/1.0"
        if proxy:
            LOCAL.session.proxies.update({"http": proxy, "https": proxy})
    lock = HOST_LOCKS[urlparse(job["url"]).hostname]
    last_error = None
    for attempt in range(20):
        delay = min(60, 3 * (attempt + 1))
        try:
            with lock:
                with LOCAL.session.get(job["url"], headers={"Range": f"bytes={start}-{end}"},
                                       stream=True, timeout=(90, 90)) as response:
                    if response.status_code in (403, 429, 503):
                        delay = max(60, int(response.headers.get("Retry-After", "60")))
                    response.raise_for_status()
                    content_range = response.headers.get("Content-Range", "")
                    if response.status_code != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                        raise ValueError(f"Server returned unexpected range {content_range}")
                    count = 0
                    with temporary.open("wb") as handle:
                        for block in response.iter_content(128 * 1024):
                            count += len(block)
                            if count > expected:
                                raise ValueError("Range exceeds requested size")
                            handle.write(block)
                            if count == expected:
                                break
                    if count != expected:
                        raise ValueError(f"Short range {count}/{expected}")
                    temporary.replace(target)
                    return count
        except (requests.RequestException, ValueError, OSError) as exc:
            last_error = str(exc)
            temporary.unlink(missing_ok=True)
            if attempt in (0, 4, 9, 19):
                announce("RETRY", job["key"], index, attempt + 1, last_error[:180])
            time.sleep(delay)
    raise RuntimeError(f"Failed range {index} in {job['key']}: {last_error}")

def complete(job, folder, path):
    expected = job.get("bytes")
    if expected and path.stat().st_size != expected:
        raise ValueError(f"Incorrect file size for {path}")
    sha, md5 = digest(path)
    if job.get("md5") and md5 != job["md5"]:
        raise ValueError(f"Publisher MD5 mismatch for {path}")
    if job.get("expected_sha256") and sha != job["expected_sha256"]:
        raise ValueError(f"Publisher SHA-256 mismatch for {path}")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC mismatch in {bad}")
        members = archive.namelist()
    (folder / "archive_members.txt").write_text("\n".join(members) + "\n")
    receipt = {**job, "actual_bytes": path.stat().st_size, "sha256": sha,
               "md5_actual": md5, "zip_crc": "passed", "members": len(members),
               "completed_utc": datetime.now(timezone.utc).isoformat()}
    announce("VERIFIED", job["key"], receipt["actual_bytes"], sha)
    return receipt

def download(job, root, workers, block_size, proxy):
    folder = (root / job["key"]).resolve()
    folder.relative_to(root)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / job["name"]
    if (folder / "receipt.json").exists():
        receipt = complete(job, folder, target)
        atomic_json(folder / "receipt.json", receipt)
        return receipt
    parts = folder / (job["name"] + ".parts")
    parts.mkdir(exist_ok=True)
    size = job["bytes"]
    count = (size + block_size - 1) // block_size
    done = 0
    stamp = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(receive_piece, job, i, i * block_size,
                               min(size - 1, (i + 1) * block_size - 1), parts, proxy)
                   for i in range(count)]
        for future in as_completed(futures):
            done += future.result()
            if time.monotonic() - stamp >= 30:
                atomic_json(folder / "progress.json", {"completed_range_bytes": done,
                            "expected_bytes": size, "at_utc": datetime.now(timezone.utc).isoformat()})
                announce("PROGRESS", job["key"], done, "/", size)
                stamp = time.monotonic()
    assembling = folder / (job["name"] + ".assembling")
    with assembling.open("wb") as output:
        for i in range(count):
            with (parts / f"{i:06}.part").open("rb") as source:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    receipt = complete(job, folder, assembling)
    assembling.replace(target)
    atomic_json(folder / "receipt.json", receipt)
    shutil.rmtree(parts)
    for suffix in (".aria2", ".incoming"):
        (folder / (job["name"] + suffix)).unlink(missing_ok=True)
    return receipt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--block-mib", type=int, default=1)
    parser.add_argument("--proxy")
    args = parser.parse_args()
    root = args.root.resolve()
    jobs = json.loads((root / "selected.json").read_text())
    for job in jobs:
        host = urlparse(job["url"]).hostname
        capacity = 12 if host == "zenodo.org" else 48 if host.endswith("amazonaws.com") else 16
        HOST_LOCKS.setdefault(host, threading.BoundedSemaphore(capacity))
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(download, job, root, args.workers, args.block_mib * 1024 * 1024,
                               args.proxy): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"key": job["key"], "error": str(exc)})
                announce("FAILED", job["key"], str(exc))
    atomic_json(root / "download_summary.json", {"completed": results, "errors": errors})
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
