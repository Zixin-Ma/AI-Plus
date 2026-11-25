"""Bundle source and video as PDF attachments for the platform's single document field."""
from pathlib import Path
import hashlib
import json
import zipfile

import pymupdf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "work/delivery"


def main():
    source = OUT / "source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(list(ROOT.glob("*.py")) + list(ROOT.glob("app/*.py")) +
                           list(ROOT.glob("*.md")) + [ROOT/"Dockerfile", ROOT/".dockerignore", ROOT/".gitignore"]):
            z.write(path, path.relative_to(ROOT))
    with pymupdf.open(OUT/"技术报告.pdf") as doc:
        doc.subset_fonts()
        doc.embfile_add("source.zip",source.read_bytes(),filename="source.zip",desc="Source, dependencies and technical references")
        doc.embfile_add("workflow.mp4",(OUT/"流程演示.mp4").read_bytes(),filename="workflow.mp4",desc="60-second workflow replay")
        doc.set_metadata({"title":"工业模型在漂移下的持续更新 | 校赛技术报告",
                          "subject":"公开实验、设计与来源；附件包含完整参赛源码与流程演示"})
        doc.save(OUT/"作品文档.pdf",garbage=4,deflate=True)
    with pymupdf.open(OUT/"作品文档.pdf") as doc:
        assert len(doc)==5 and doc.embfile_get("source.zip")==source.read_bytes()
        assert doc.embfile_get("workflow.mp4")==(OUT/"流程演示.mp4").read_bytes()
    paths = ("harness.tar","作品文档.pdf","技术报告.pdf","流程演示.mp4","source.zip")
    manifest={n:{"bytes":(OUT/n).stat().st_size, "sha256":hashlib.sha256((OUT/n).read_bytes()).hexdigest()} for n in paths}
    (OUT/"SHA256.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
