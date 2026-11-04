"""Download the model weights listed in manifest.json into DEST/<model>/ and verify size + sha256 (run at image build;
the build fails on any mismatch, so a changed or truncated file never ends up in the image).
usage: python fetch_weights.py MANIFEST DEST"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def fetch(url: str, dest: Path, sha256: str, size: int):
    h, n = hashlib.sha256(), 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
    if n != size or h.hexdigest() != sha256:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"{dest.name}: got {n} bytes sha256 {h.hexdigest()}, the manifest says {size} / {sha256}")
    print(f"ok {dest.name} {n} bytes {sha256[:12]}")


def main(manifest: str, dest: str):
    models = json.loads(Path(manifest).read_text(encoding="utf-8"))["models"]
    for name, m in models.items():
        for f in m["files"]:
            fetch(f["url"], Path(dest) / name / f["path"], f["sha256"], f["bytes"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
