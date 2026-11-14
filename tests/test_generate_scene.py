import json

import numpy as np
from PIL import Image

import config
import image_generator as ig
from conftest import make_story


class FakePipeline:
    tokenizer = None

    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        class R: images = [Image.new("RGB", (512, 768), (30, 60, 90))]
        return R()


def job():
    import scene_processor
    return scene_processor.prepare_scene_jobs(make_story())[0]


def test_generates_cropped_image_and_metadata(tmp_path):
    pipe = FakePipeline()
    out = ig.generate_scene_image(pipe, job(), image_dir=tmp_path)
    assert Image.open(out).size == (432, 768)
    meta = json.loads((tmp_path / "scene_01.json").read_text(encoding="utf-8"))
    assert meta["seed"] == config.BASE_SEED + 1
    assert meta["prompt"] == pipe.calls[0]["prompt"]
    assert pipe.calls[0]["width"] == 512 and pipe.calls[0]["height"] == 768


def test_regenerating_backs_up_the_old_image(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BACKUP_DIR", tmp_path / "backups")
    ig.generate_scene_image(FakePipeline(), job(), image_dir=tmp_path)
    first = (tmp_path / "scene_01.png").read_bytes()
    pipe = FakePipeline()
    pipe_image = Image.new("RGB", (512, 768), (200, 10, 10))
    pipe.__class__ = type("P2", (FakePipeline,), {"__call__": lambda s, **kw: type("R", (), {"images": [pipe_image]})()})
    ig.generate_scene_image(pipe, job(), image_dir=tmp_path)
    backups = list((tmp_path / "backups").glob("scene_01.*.png"))
    assert len(backups) == 1 and backups[0].read_bytes() == first
    assert (tmp_path / "scene_01.png").read_bytes() != first


def test_refiner_is_applied_before_cropping(tmp_path):
    seen = {}
    def refiner(image, seed):
        seen["size"] = image.size
        return Image.new("RGB", image.size, (255, 255, 255)), [{"face": [1, 2, 3, 4]}]
    out = ig.generate_scene_image(FakePipeline(), job(), refiner=refiner, image_dir=tmp_path)
    assert seen["size"] == (512, 768)
    assert Image.open(out).getpixel((10, 10)) == (255, 255, 255)
    meta = json.loads((tmp_path / "scene_01.json").read_text(encoding="utf-8"))
    assert meta["face_refinement"] == [{"face": [1, 2, 3, 4]}]


def test_missing_model_without_download_is_a_clear_error(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "nomodel")
    with pytest.raises(ig.ModelNotInstalledError):
        ig.load_image_pipeline("CPU", allow_download=False)
