import time

import benchmark as bm


def test_cpu_gpu_auto_selection():
    d = bm.select_devices(["CPU", "GPU", "AUTO"], ["CPU", "GPU.0"])
    assert [x[1] for x in d] == [True, True, True]


def test_gpu_not_assumed_when_not_reported():
    d = dict((x[0], x) for x in bm.select_devices(["CPU", "GPU"], ["CPU"]))
    assert d["CPU"][1] and not d["GPU"][1] and "no GPU" in d["GPU"][2]


def test_everything_skipped_without_openvino():
    assert not any(r for _, r, _ in bm.select_devices(["CPU", "AUTO"], None))


def test_ollama_metrics():
    m = bm.ollama_metrics({"eval_count": 100, "eval_duration": 5_000_000_000,
                           "prompt_eval_count": 40, "prompt_eval_duration": 2_000_000_000,
                           "load_duration": 1_500_000_000, "total_duration": 9_000_000_000})
    assert m["generated_tokens_per_second"] == 20.0 and m["prompt_tokens_per_second"] == 20.0
    assert m["load_seconds"] == 1.5


def test_ollama_failure_is_reported_not_raised():
    import urllib.error
    def boom(*a, **k): raise urllib.error.URLError("refused")
    assert bm.run_ollama_benchmark(runs=1, opener=boom)["status"] == "failed"


class FakePipe:
    def __call__(self, **kw):
        time.sleep(0.01)


def test_worker_times_images_and_excludes_warmup():
    r = bm.run_worker("CPU", images=2, steps=10, warmup=1, loader=lambda d: FakePipe())
    assert r["status"] == "ok" and len(r["image_seconds"]) == 2
    assert r["first_image_seconds"] is not None and r["seconds_per_step"] is not None


def test_worker_flags_a_second_model_copy():
    big = lambda pipe: [bytearray(1)]  # builder that allocates nothing
    r = bm.run_worker("CPU", 1, 5, 0, loader=lambda d: FakePipe(), check_refiner_memory=True, refiner_builder=big)
    assert r["refiner"]["built"] and r["refiner"]["second_copy_suspected"] is False


def test_worker_reports_refiner_failure():
    def fail(pipe): raise RuntimeError("not shareable")
    r = bm.run_worker("CPU", 1, 5, 0, loader=lambda d: FakePipe(), check_refiner_memory=True, refiner_builder=fail)
    assert r["refiner"] == {"built": False, "reason": "not shareable"}


def test_markdown_report_has_a_row_per_device():
    report = {"created": "now", "system": {"platform": "Win", "processor": "x", "ram_gb": 16, "python": "3.13"},
              "openvino": {"version": "2025", "device_names": {"CPU": "Core Ultra 5"}},
              "settings": {"steps": 20, "images": 2, "warmup": 1},
              "devices": [{"device": "CPU", "status": "ok", "mean_image_seconds": 90.0},
                          {"device": "GPU", "status": "skipped", "reason": "no GPU"}]}
    md = bm.render_markdown(report)
    assert "| CPU | ok |" in md and "| GPU | skipped |" in md and "Core Ultra 5" in md


def test_peak_rss_sampler_runs():
    t = bm.PeakRSS(interval=0.01); t.start(); time.sleep(0.05); t.stop()
    assert t.peak_mb is None or t.peak_mb > 0
