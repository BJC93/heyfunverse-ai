"""Environment doctor: checks that this machine can run HeyFunverse AI.

    python ai-engine/check_env.py
    python ai-engine/check_env.py --json
    python ai-engine/check_env.py --probe-face       # load face models, measure memory
    python ai-engine/check_env.py --probe-refiner    # read-only OpenVINO signature check

Exit code 0 = nothing failed, 1 = at least one required check failed.
A device listed by OpenVINO is NOT proof that it works for image generation:
run benchmark.py to find out.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

import config

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"


@dataclass
class Check:
    name: str
    status: str
    detail: str


# ------------------------------------------------------------------
# Individual checks (each returns a list of Check)
# ------------------------------------------------------------------

def check_python():

    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro} ({platform.system()})"

    if version < (3, 10):
        return [Check("Python", FAIL, text + ": 3.10 or newer is required")]

    return [Check("Python", OK, text)]


REQUIRED_PACKAGES = [
    # (display name, pip distribution, import name)
    ("numpy", "numpy", "numpy"),
    ("Pillow", "pillow", "PIL"),
    ("OpenVINO", "openvino", "openvino"),
    ("optimum-intel", "optimum-intel", "optimum.intel"),
    ("diffusers", "diffusers", "diffusers"),
    ("transformers", "transformers", "transformers"),
    ("torch", "torch", "torch"),
    ("psutil", "psutil", "psutil"),
]

OPTIONAL_PACKAGES = [
    ("OpenCV", "opencv-python", "cv2"),
]


def _package_version(distribution, import_name):

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass

    try:
        module = importlib.import_module(import_name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def check_packages():

    results = []

    for label, dist, module in REQUIRED_PACKAGES:
        version = _package_version(dist, module)

        if version:
            results.append(Check(label, OK, str(version)))
        else:
            results.append(Check(label, FAIL, "not installed (pip install -r requirements.txt)"))

    for label, dist, module in OPTIONAL_PACKAGES:
        version = _package_version(dist, module)

        if version:
            results.append(Check(label, OK, f"{version} (optional, face tools)"))
        else:
            results.append(Check(label, INFO, "not installed (optional, needed for face tools)"))

    return results


def check_openvino_devices():

    try:
        import openvino
    except ImportError:
        return [Check("OpenVINO devices", FAIL, "OpenVINO is not installed")]

    try:
        core = openvino.Core()
        devices = list(core.available_devices)
    except Exception as error:
        return [Check("OpenVINO devices", FAIL, f"could not query devices: {error}")]

    results = []

    for device in devices:
        try:
            name = core.get_property(device, "FULL_DEVICE_NAME")
        except Exception:
            name = "name unavailable"

        results.append(Check(f"Device {device}", INFO, str(name)))

    if not devices:
        results.append(Check("OpenVINO devices", FAIL, "no devices reported"))
    else:
        results.append(Check(
            "Device note", INFO,
            "listed does not mean working: run benchmark.py before using GPU or AUTO"
        ))

    return results


def parse_ffmpeg_buildconf(text):
    """Pick the features we care about out of ``ffmpeg -buildconf`` output."""

    return {
        "libass": "--enable-libass" in text,
        "harfbuzz": "--enable-libharfbuzz" in text,
        "fribidi": "--enable-libfribidi" in text,
        "x264": "--enable-libx264" in text,
    }


def check_ffmpeg(run=subprocess.run, which=shutil.which):

    ffmpeg = which("ffmpeg")

    if not ffmpeg:
        return [Check("FFmpeg", FAIL, "ffmpeg not found on PATH (needed from Phase 3)")]

    def call(*args):
        completed = run(
            [ffmpeg, *args], capture_output=True, text=True, timeout=30
        )
        return (completed.stdout or "") + (completed.stderr or "")

    results = []

    try:
        first_line = call("-version").splitlines()[0]
        results.append(Check("FFmpeg", OK, first_line))

        features = parse_ffmpeg_buildconf(call("-hide_banner", "-buildconf"))

        for key, label in (
            ("libass", "FFmpeg libass (subtitles)"),
            ("harfbuzz", "FFmpeg HarfBuzz (Telugu shaping)"),
            ("x264", "FFmpeg libx264 (H.264 encoding)"),
        ):
            status = OK if features[key] else WARN
            results.append(Check(label, status, "present" if features[key] else "missing"))

    except Exception as error:
        results.append(Check("FFmpeg", WARN, f"found but could not be queried: {error}"))

    return results


def check_ollama(opener=urllib.request.urlopen):

    try:
        with opener(config.OLLAMA_TAGS_URL, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))

    except (urllib.error.URLError, OSError, ValueError) as error:
        return [Check(
            "Ollama", WARN,
            f"not reachable at {config.OLLAMA_BASE_URL} ({error}); "
            "only needed for local story planning"
        )]

    names = [m.get("name", "") for m in data.get("models", [])]

    if any(n == config.OLLAMA_MODEL or n.startswith(config.OLLAMA_MODEL + ":") for n in names):
        return [Check("Ollama", OK, f"running, model {config.OLLAMA_MODEL} available")]

    return [Check(
        "Ollama", WARN,
        f"running, but model {config.OLLAMA_MODEL} not found "
        f"(run: ollama pull {config.OLLAMA_MODEL})"
    )]


def check_resources():

    results = []

    try:
        import psutil

        memory = psutil.virtual_memory()
        total = memory.total / 1024 ** 3
        free = memory.available / 1024 ** 3
        status = OK if total >= 14 else WARN
        results.append(Check("RAM", status, f"{total:.1f} GB total, {free:.1f} GB free now"))

        results.append(Check(
            "CPU", INFO,
            f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads"
        ))
    except ImportError:
        results.append(Check("RAM", INFO, "psutil missing, memory not checked"))

    try:
        target = config.STORAGE_DIR if config.STORAGE_DIR.exists() else config.PROJECT_ROOT
        free_gb = shutil.disk_usage(target).free / 1024 ** 3
        status = OK if free_gb >= 15 else WARN
        results.append(Check("Disk space", status, f"{free_gb:.1f} GB free (15 GB or more recommended)"))
    except OSError as error:
        results.append(Check("Disk space", WARN, str(error)))

    return results


def check_assets():

    model_ready = (config.MODEL_DIR / "model_index.json").exists()

    results = [Check(
        "Image model",
        OK if model_ready else INFO,
        str(config.MODEL_DIR) if model_ready
        else "not converted yet: the first image run downloads and converts it (internet needed)"
    )]

    for label, path in (
        ("YuNet face model", config.YUNET_MODEL_FILE),
        ("SFace face model", config.SFACE_MODEL_FILE),
    ):
        results.append(Check(
            label, OK if path.is_file() else INFO,
            "present" if path.is_file() else "optional, not installed (see docs/FACE_MODELS.md)"
        ))

    fonts = sorted(config.FONT_DIR.glob("NotoSansTelugu*")) if config.FONT_DIR.exists() else []

    results.append(Check(
        "Telugu font",
        OK if fonts else INFO,
        fonts[0].name if fonts else "Noto Sans Telugu not installed yet (needed from Phase 3)"
    ))

    return results


# ------------------------------------------------------------------
# Optional probes
# ------------------------------------------------------------------

def probe_face_models():
    """Load the optional face models and measure the memory they add."""

    try:
        import cv2
        import numpy as np
        import psutil
    except ImportError as error:
        return [Check("Face probe", WARN, f"needs opencv-python and psutil: {error}")]

    process = psutil.Process()
    results = []
    before = process.memory_info().rss

    if config.YUNET_MODEL_FILE.is_file():
        try:
            detector = cv2.FaceDetectorYN.create(
                str(config.YUNET_MODEL_FILE), "", (320, 320), 0.7, 0.3, 50
            )
            detector.setInputSize((512, 768))
            detector.detect(np.zeros((768, 512, 3), dtype=np.uint8))
            delta = (process.memory_info().rss - before) / 1024 ** 2
            results.append(Check("YuNet probe", OK, f"loads and runs on OpenCV {cv2.__version__}; +{delta:.0f} MB RAM"))
        except Exception as error:
            results.append(Check("YuNet probe", FAIL, f"could not run: {error}"))
    else:
        results.append(Check("YuNet probe", INFO, "model file not installed"))

    before = process.memory_info().rss

    if config.SFACE_MODEL_FILE.is_file():
        try:
            recognizer = cv2.FaceRecognizerSF.create(str(config.SFACE_MODEL_FILE), "")
            recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8))
            delta = (process.memory_info().rss - before) / 1024 ** 2
            results.append(Check("SFace probe", OK, f"loads and runs on OpenCV {cv2.__version__}; +{delta:.0f} MB RAM"))
        except Exception as error:
            results.append(Check("SFace probe", FAIL, f"could not run: {error}"))
    else:
        results.append(Check("SFace probe", INFO, "model file not installed"))

    return results


def probe_refiner():
    """Read-only: show how the img2img pipeline's constructor compares with
    the components of the text-to-image pipeline. Loads no model."""

    try:
        from optimum.intel import (
            OVStableDiffusionImg2ImgPipeline,
            OVStableDiffusionPipeline,
        )
        import inspect
    except ImportError as error:
        return [Check("Refiner probe", WARN, f"optimum-intel not installed: {error}")]

    img2img = inspect.signature(OVStableDiffusionImg2ImgPipeline.__init__)
    text2img = inspect.signature(OVStableDiffusionPipeline.__init__)

    needed = [
        n for n, p in img2img.parameters.items()
        if n != "self"
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and p.default is p.empty
    ]
    provided = set(text2img.parameters)
    unmatched = [n for n in needed if n not in provided]

    return [
        Check("img2img signature", INFO, str(img2img)),
        Check("text2img signature", INFO, str(text2img)),
        Check(
            "Refiner probe",
            OK if not unmatched else WARN,
            "required img2img parameters all exist on the text2img pipeline"
            if not unmatched else
            "img2img needs parameters the text2img pipeline lacks: " + ", ".join(unmatched)
        ),
    ]


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_checks(probe_face=False, probe_refiner_flag=False):

    results = []

    for step in (
        check_python,
        check_packages,
        check_openvino_devices,
        check_ffmpeg,
        check_ollama,
        check_resources,
        check_assets,
    ):
        results.extend(step())

    if probe_face:
        results.extend(probe_face_models())

    if probe_refiner_flag:
        results.extend(probe_refiner())

    return results


def exit_code(results):
    return 1 if any(c.status == FAIL for c in results) else 0


def format_report(results):

    symbols = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]", INFO: "[info]"}
    width = max(len(c.name) for c in results)

    lines = [f"{symbols[c.status]} {c.name.ljust(width)}  {c.detail}" for c in results]

    failed = sum(c.status == FAIL for c in results)
    warned = sum(c.status == WARN for c in results)

    lines.append("")
    lines.append(f"{failed} failed, {warned} warnings.")

    return "\n".join(lines)


def main(argv=None):

    parser = argparse.ArgumentParser(description="HeyFunverse AI environment check")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--probe-face", action="store_true", help="load face models and measure memory")
    parser.add_argument("--probe-refiner", action="store_true", help="compare OpenVINO pipeline signatures")
    args = parser.parse_args(argv)

    results = run_checks(args.probe_face, args.probe_refiner)

    if args.json:
        print(json.dumps([asdict(c) for c in results], indent=2))
    else:
        print(format_report(results))

    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
