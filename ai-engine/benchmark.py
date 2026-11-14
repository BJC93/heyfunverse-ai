"""CPU / GPU / AUTO benchmark for image generation, plus an optional Ollama test.

    python ai-engine/benchmark.py                          # CPU only (safe default)
    python ai-engine/benchmark.py --devices CPU GPU AUTO --images 2 --steps 20
    python ai-engine/benchmark.py --ollama                 # Qwen tokens per second
    python ai-engine/benchmark.py --check-refiner-memory   # proves no second model copy

Each device runs in its own child process, so a GPU driver crash or an
out-of-memory error only fails that one row. Nothing is downloaded: the
converted image model must already exist (generate one image first).
Results are written to storage/benchmarks/ as JSON and Markdown.

For fair numbers: plug in the charger, set the Windows power mode to best
performance, and close browsers and other heavy apps. Do not run the Ollama
test and the image test at the same time.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import config
import safe_io

DEFAULT_PROMPT = (
    "portrait of a young Indian woman in a pastel blue saree, rainy evening "
    "street, soft street lights, RAW photo, photorealistic, 85mm"
)


# ------------------------------------------------------------------
# Memory sampling
# ------------------------------------------------------------------

class PeakRSS:
    """Samples this process's resident memory in a background thread."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil
            self._process = psutil.Process()
        except ImportError:
            self._process = None

    @property
    def available(self):
        return self._process is not None

    def current_bytes(self):
        return self._process.memory_info().rss if self._process else 0

    def _loop(self):
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self.current_bytes())
            self._stop.wait(self.interval)

    def start(self):
        if self._process:
            self.peak_bytes = self.current_bytes()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._process:
            self.peak_bytes = max(self.peak_bytes, self.current_bytes())

    @property
    def peak_mb(self):
        return round(self.peak_bytes / 1024 ** 2) if self._process else None


# ------------------------------------------------------------------
# Device selection
# ------------------------------------------------------------------

def select_devices(requested, available):
    """Decide which requested devices can be attempted.

    Returns [(device, runnable, note)]. ``available`` is OpenVINO's device
    list, or None if OpenVINO is not installed."""

    decisions = []

    for device in requested:

        if available is None:
            decisions.append((device, False, "OpenVINO is not installed"))

        elif device == "CPU":
            ok = "CPU" in available
            decisions.append((device, ok, "" if ok else "OpenVINO does not report a CPU device"))

        elif device == "GPU":
            ok = any(d.startswith("GPU") for d in available)
            decisions.append((device, ok, "" if ok else
                              "OpenVINO reports no GPU device (check Intel graphics drivers)"))

        elif device == "AUTO":
            ok = bool(available)
            decisions.append((device, ok, "" if ok else "no devices available"))

        else:
            decisions.append((device, False, "unsupported device name"))

    return decisions


def openvino_info():

    try:
        import openvino
    except ImportError:
        return None, {}

    core = openvino.Core()
    devices = list(core.available_devices)
    names = {}

    for device in devices:
        try:
            names[device] = str(core.get_property(device, "FULL_DEVICE_NAME"))
        except Exception:
            names[device] = "unknown"

    return devices, {
        "version": getattr(openvino, "__version__", "unknown"),
        "device_names": names,
    }


def system_info():

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        info["ram_gb"] = None

    return info


# ------------------------------------------------------------------
# Worker: runs inside a child process for one device
# ------------------------------------------------------------------

def run_worker(
    device,
    images,
    steps,
    warmup=1,
    prompt=DEFAULT_PROMPT,
    loader=None,
    check_refiner_memory=False,
    refiner_builder=None
):
    """Load the pipeline on ``device`` and time image generation.

    ``loader`` and ``refiner_builder`` can be replaced in tests."""

    import numpy as np

    if loader is None:
        import image_generator

        def loader(dev):
            return image_generator.load_image_pipeline(dev, allow_download=False)

    tracker = PeakRSS()
    tracker.start()

    started = time.perf_counter()
    pipeline = loader(device)
    load_seconds = time.perf_counter() - started

    after_load_mb = round(tracker.current_bytes() / 1024 ** 2) if tracker.available else None

    timings = []

    for index in range(warmup + images):

        began = time.perf_counter()

        pipeline(
            prompt=prompt,
            negative_prompt=config.NEGATIVE_PROMPT,
            width=config.WIDTH,
            height=config.HEIGHT,
            num_inference_steps=steps,
            guidance_scale=config.GUIDANCE_SCALE,
            num_images_per_prompt=1,
            generator=np.random.RandomState(config.BASE_SEED + index)
        )

        timings.append(time.perf_counter() - began)

    refiner_report = None

    if check_refiner_memory:

        if refiner_builder is None:
            import face_refine
            refiner_builder = face_refine.build_shared_img2img

        before = tracker.current_bytes()

        try:
            refiner_builder(pipeline)
            grew_mb = round((tracker.current_bytes() - before) / 1024 ** 2)
            refiner_report = {
                "built": True,
                "ram_growth_mb": grew_mb,
                # A second SD1.5 copy would add well over a gigabyte.
                "second_copy_suspected": grew_mb > 1024,
            }
        except Exception as error:
            refiner_report = {"built": False, "reason": str(error)}

    tracker.stop()

    first = timings[0] if warmup else None
    measured = timings[warmup:]

    return {
        "device": device,
        "status": "ok",
        "load_seconds": round(load_seconds, 1),
        "ram_after_load_mb": after_load_mb,
        "first_image_seconds": round(first, 1) if first is not None else None,
        "image_seconds": [round(t, 1) for t in measured],
        "mean_image_seconds": round(statistics.mean(measured), 1) if measured else None,
        "seconds_per_step": round(statistics.mean(measured) / steps, 2) if measured else None,
        "peak_ram_mb": tracker.peak_mb,
        "steps": steps,
        "size": [config.WIDTH, config.HEIGHT],
        "refiner": refiner_report,
    }


# ------------------------------------------------------------------
# Ollama
# ------------------------------------------------------------------

def ollama_metrics(response):
    """Turn an Ollama /api/generate response into readable numbers."""

    def seconds(key):
        return response.get(key, 0) / 1e9

    eval_seconds = seconds("eval_duration")
    prompt_seconds = seconds("prompt_eval_duration")

    return {
        "load_seconds": round(seconds("load_duration"), 2),
        "prompt_tokens": response.get("prompt_eval_count"),
        "prompt_tokens_per_second": (
            round(response.get("prompt_eval_count", 0) / prompt_seconds, 1)
            if prompt_seconds else None
        ),
        "generated_tokens": response.get("eval_count"),
        "generated_tokens_per_second": (
            round(response.get("eval_count", 0) / eval_seconds, 1)
            if eval_seconds else None
        ),
        "total_seconds": round(seconds("total_duration"), 2),
    }


def run_ollama_benchmark(runs=2, num_predict=128, opener=urllib.request.urlopen):

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": "Write a short romantic story outline in four numbered scenes.",
        "stream": False,
        "keep_alive": 0,  # unload afterwards so it does not hold RAM
        "options": {"temperature": 0.1, "num_predict": num_predict},
    }

    results = []

    for _ in range(runs):

        request = urllib.request.Request(
            config.OLLAMA_GENERATE_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with opener(request, timeout=config.OLLAMA_TIMEOUT_SECONDS) as response:
                results.append(ollama_metrics(json.loads(response.read().decode("utf-8"))))
        except (urllib.error.URLError, OSError, ValueError) as error:
            return {"status": "failed", "reason": str(error), "model": config.OLLAMA_MODEL}

    return {"status": "ok", "model": config.OLLAMA_MODEL, "runs": results}


# ------------------------------------------------------------------
# Parent: orchestrates child processes and writes the report
# ------------------------------------------------------------------

def run_device_in_child(device, args, timeout):

    with tempfile.TemporaryDirectory() as folder:

        out_file = Path(folder) / "result.json"

        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--device", device,
            "--images", str(args.images),
            "--steps", str(args.steps),
            "--warmup", str(args.warmup),
            "--out", str(out_file),
        ]

        if args.check_refiner_memory:
            command.append("--check-refiner-memory")

        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return {"device": device, "status": "timeout",
                    "reason": f"no result after {timeout} s"}

        if out_file.exists():
            return json.loads(out_file.read_text(encoding="utf-8"))

        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-6:]

        return {
            "device": device,
            "status": "failed",
            "reason": " | ".join(tail) or f"exit code {completed.returncode}",
        }


def render_markdown(report):

    lines = [
        "# HeyFunverse AI benchmark",
        "",
        f"- Date: {report['created']}",
        f"- Machine: {report['system'].get('platform')} "
        f"({report['system'].get('processor')}), "
        f"{report['system'].get('ram_gb')} GB RAM",
        f"- Python {report['system'].get('python')}, OpenVINO "
        f"{report['openvino'].get('version', 'not installed')}",
        f"- Image size {config.WIDTH}x{config.HEIGHT}, "
        f"{report['settings']['steps']} steps, "
        f"{report['settings']['images']} timed images per device",
        "",
        "| Device | Status | Load and compile (s) | First image (s) | Mean image (s) "
        "| Seconds per step | Peak RAM (MB) | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    names = report["openvino"].get("device_names", {})

    for row in report["devices"]:

        note = row.get("reason", "")

        if row["device"] in names:
            note = (names[row["device"]] + ("; " + note if note else ""))

        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["device"],
            row.get("status"),
            row.get("load_seconds", "-"),
            row.get("first_image_seconds", "-"),
            row.get("mean_image_seconds", "-"),
            row.get("seconds_per_step", "-"),
            row.get("peak_ram_mb", "-"),
            note or "-",
        ))

        refiner = row.get("refiner")
        if refiner:
            lines.append(
                f"| {row['device']} refiner | "
                + ("built" if refiner.get("built") else "not built")
                + f" | | | | | | RAM growth {refiner.get('ram_growth_mb', '-')} MB"
                + (" (SECOND COPY SUSPECTED)" if refiner.get("second_copy_suspected") else "")
                + (f"; {refiner['reason']}" if refiner.get("reason") else "")
                + " |"
            )

    ollama = report.get("ollama")

    if ollama:
        lines += ["", f"## Ollama ({ollama.get('model')})", ""]

        if ollama["status"] != "ok":
            lines.append(f"Failed: {ollama.get('reason')}")
        else:
            lines += ["| Run | Load (s) | Prompt tokens/s | Generated tokens/s | Total (s) |",
                      "| --- | --- | --- | --- | --- |"]
            for number, run in enumerate(ollama["runs"], 1):
                lines.append(
                    f"| {number} | {run['load_seconds']} | {run['prompt_tokens_per_second']} "
                    f"| {run['generated_tokens_per_second']} | {run['total_seconds']} |"
                )

    lines += ["", "Run 1 of Ollama includes model load time. A listed device is not proof "
              "that it works; this table is."]

    return "\n".join(lines) + "\n"


def run_benchmark(args):

    available, ov_info = openvino_info()

    report = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "system": system_info(),
        "openvino": ov_info,
        "settings": {"steps": args.steps, "images": args.images, "warmup": args.warmup},
        "devices": [],
    }

    if not args.skip_image:

        for device, runnable, note in select_devices(args.devices, available):

            if not runnable:
                print(f"Skipping {device}: {note}")
                report["devices"].append({"device": device, "status": "skipped", "reason": note})
                continue

            print(f"\nBenchmarking {device} (this can take several minutes)...")
            result = run_device_in_child(device, args, args.timeout)
            print(f"  -> {result.get('status')}"
                  + (f": mean {result.get('mean_image_seconds')} s/image"
                     if result.get("status") == "ok" else f": {result.get('reason')}"))
            report["devices"].append(result)

    if args.ollama:
        print("\nBenchmarking Ollama...")
        report["ollama"] = run_ollama_benchmark()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    json_path = config.BENCHMARK_DIR / f"benchmark_{stamp}.json"
    md_path = config.BENCHMARK_DIR / f"benchmark_{stamp}.md"

    safe_io.write_json_safely(json_path, report)
    safe_io.write_text_safely(md_path, render_markdown(report))

    print("\n" + render_markdown(report))
    print(f"Saved: {md_path}")

    return report


def build_parser():

    parser = argparse.ArgumentParser(description="HeyFunverse AI benchmark")
    parser.add_argument("--devices", nargs="+", default=["CPU"],
                        choices=list(config.VALID_DEVICES))
    parser.add_argument("--images", type=int, default=2, help="timed images per device")
    parser.add_argument("--steps", type=int, default=20, help="inference steps per image")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warm-up images")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed per device")
    parser.add_argument("--ollama", action="store_true", help="also benchmark Qwen via Ollama")
    parser.add_argument("--skip-image", action="store_true", help="skip image benchmarks")
    parser.add_argument("--check-refiner-memory", action="store_true",
                        help="build the shared face refiner and report RAM growth")
    # internal
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="CPU", help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv=None):

    args = build_parser().parse_args(argv)

    if args.worker:
        result = run_worker(
            args.device, args.images, args.steps, args.warmup,
            check_refiner_memory=args.check_refiner_memory
        )
        Path(args.out).write_text(json.dumps(result), encoding="utf-8")
        return 0

    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
