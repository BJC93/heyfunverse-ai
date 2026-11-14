"""Download the two small optional face models (YuNet and SFace).

Nothing is downloaded unless you pass --yes. Sizes below come from the
Hugging Face file listing of opencv/opencv_zoo; the script refuses a file
whose size is far from the expected one, and records SHA-256 hashes in
storage/models/face/SHA256SUMS.txt so you can verify them later.

    python ai-engine/fetch_face_models.py             # shows what it would do
    python ai-engine/fetch_face_models.py --yes
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import config
import safe_io

BASE = "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/"

# name, url, destination, (min_bytes, max_bytes), licence
MODELS = [
    (
        "YuNet 2023mar (face detector)",
        BASE + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        config.YUNET_MODEL_FILE,
        (150_000, 400_000),          # listed as about 233 kB
        "MIT",
    ),
    (
        "SFace 2021dec (face recogniser)",
        BASE + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        config.SFACE_MODEL_FILE,
        (36_000_000, 41_000_000),    # listed as 38.7 MB
        "Apache 2.0",
    ),
]


def size_ok(size, bounds):
    low, high = bounds
    return low <= size <= high


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination, bounds, opener=urllib.request.urlopen):
    """Download to a .part file, check the size, then move into place.
    An existing file is backed up first."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")

    with opener(url, timeout=120) as response, open(part, "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    size = part.stat().st_size

    if not size_ok(size, bounds):
        part.unlink()
        raise ValueError(
            f"Downloaded {size} bytes, expected between {bounds[0]} and {bounds[1]}. "
            "File discarded."
        )

    safe_io.backup_file(destination)
    part.replace(destination)

    return size, sha256_of(destination)


def main(argv=None):

    parser = argparse.ArgumentParser(description="Fetch optional face models")
    parser.add_argument("--yes", action="store_true", help="actually download")
    args = parser.parse_args(argv)

    for name, url, destination, bounds, licence in MODELS:
        print(f"{name}\n  licence {licence}\n  from {url}\n  to   {destination}")

    if not args.yes:
        print("\nDry run. Add --yes to download (about 39 MB in total).")
        return 0

    lines = []

    for name, url, destination, bounds, licence in MODELS:
        print(f"\nDownloading {name}...")
        size, digest = download(url, destination, bounds)
        print(f"  {size:,} bytes, sha256 {digest}")
        lines.append(f"{digest}  {Path(destination).name}")

    safe_io.write_text_safely(
        config.FACE_MODEL_DIR / "SHA256SUMS.txt", "\n".join(lines) + "\n"
    )

    print("\nDone. Verify with: python ai-engine/check_env.py --probe-face")
    return 0


if __name__ == "__main__":
    sys.exit(main())
