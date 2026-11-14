import numpy as np
import pytest
from PIL import Image

import face_refine as fr


class FakeDetector:
    name = "fake"

    def __init__(self, faces):
        self.faces = faces

    def detect(self, image):
        return self.faces


@pytest.mark.parametrize("face", [
    fr.FaceBox(200, 150, 80, 90),
    fr.FaceBox(0, 0, 60, 60),
    fr.FaceBox(450, 700, 60, 60),
    fr.FaceBox(100, 300, 300, 300),
])
def test_crop_box_inside_image_with_right_aspect(face):
    left, top, right, bottom = fr.crop_box_for_face(face, (512, 768), (512, 768))
    assert 0 <= left < right <= 512 and 0 <= top < bottom <= 768
    assert abs((right - left) / (bottom - top) - 512 / 768) < 0.01


def test_crop_contains_a_typical_face():
    face = fr.FaceBox(200, 150, 80, 90)
    left, top, right, bottom = fr.crop_box_for_face(face, (512, 768), (512, 768))
    assert left <= face.x and top <= face.y
    assert right >= face.x + face.w and bottom >= face.y + face.h


def test_feather_mask_is_soft_oval():
    mask = fr.feather_mask((200, 300))
    assert mask.getpixel((100, 150)) == 255
    assert mask.getpixel((0, 0)) == 0


def test_refine_changes_only_near_the_face():
    base = Image.new("RGB", (512, 768), (10, 10, 10))
    calls = []

    def img2img(**kw):
        calls.append(kw)
        return Image.new("RGB", kw["image"].size, (250, 250, 250))

    out, report = fr.refine_faces(
        base, FakeDetector([fr.FaceBox(200, 200, 80, 90)]), img2img,
        prompt="p", negative_prompt="n", out_size=(512, 768),
        strength=0.35, steps=5, seed=1,
    )
    assert len(calls) == 1 and calls[0]["strength"] == 0.35
    assert calls[0]["image"].size == (512, 768)
    assert out.getpixel((240, 245)) == (250, 250, 250)     # face centre refined
    assert out.getpixel((5, 5)) == (10, 10, 10)            # far corner untouched
    assert base.getpixel((240, 245)) == (10, 10, 10)       # input not modified
    assert report[0]["detector"] == "fake"


def test_small_faces_are_skipped_and_max_faces_respected():
    base = Image.new("RGB", (512, 768))
    seen = []
    img2img = lambda **kw: (seen.append(1), kw["image"])[1]
    faces = [fr.FaceBox(10, 10, 20, 20), fr.FaceBox(100, 100, 80, 80),
             fr.FaceBox(300, 300, 90, 90), fr.FaceBox(200, 500, 70, 70)]
    _, report = fr.refine_faces(base, FakeDetector(faces), img2img, prompt="p",
                                negative_prompt="n", out_size=(512, 768), strength=0.3,
                                steps=1, seed=0, max_faces=2, min_face=40)
    assert len(report) == 2 and len(seen) == 2


def test_no_faces_returns_unchanged_copy():
    base = Image.new("RGB", (64, 96), (1, 2, 3))
    out, report = fr.refine_faces(base, FakeDetector([]), lambda **k: None, prompt="p",
                                  negative_prompt="n", out_size=(64, 96), strength=0.3, steps=1, seed=0)
    assert report == [] and out.tobytes() == base.tobytes()


# ---- sharing the loaded pipeline: never load a second copy ----

class FakeLoaded:
    unet = object()
    vae_encoder = object()
    vae_decoder = object()
    text_encoder = object()
    tokenizer = object()
    scheduler = object()
    config = {"a": 1}


class FakeImg2Img:
    from_pretrained_calls = 0
    constructed = []

    def __init__(self, unet, vae_encoder, vae_decoder, text_encoder, tokenizer, scheduler, config=None, ov_config=None):
        FakeImg2Img.constructed.append((unet, vae_encoder))

    @classmethod
    def from_pretrained(cls, *a, **k):
        cls.from_pretrained_calls += 1
        raise AssertionError("a second model copy must never be loaded")


def test_shared_pipeline_reuses_the_same_component_objects():
    FakeImg2Img.constructed.clear()
    loaded = FakeLoaded()
    fr.build_shared_img2img(loaded, FakeImg2Img)
    unet, encoder = FakeImg2Img.constructed[0]
    assert unet is loaded.unet and encoder is loaded.vae_encoder
    assert FakeImg2Img.from_pretrained_calls == 0


def test_missing_component_disables_refinement_instead_of_loading_more():
    class NoEncoder(FakeLoaded):
        vae_encoder = None
    with pytest.raises(fr.RefinerUnavailable):
        fr.build_shared_img2img(NoEncoder(), FakeImg2Img)


def test_unmatched_required_parameter_is_reported():
    class Odd:
        def __init__(self, unet, something_new): ...
    with pytest.raises(fr.RefinerUnavailable, match="something_new"):
        fr.map_components(Odd, FakeLoaded())


def test_constructor_failure_becomes_unavailable():
    class Boom:
        def __init__(self, unet): raise RuntimeError("nope")
    with pytest.raises(fr.RefinerUnavailable):
        fr.build_shared_img2img(FakeLoaded(), Boom)


def test_describe_sharing_is_read_only():
    FakeImg2Img.constructed.clear()
    info = fr.describe_sharing(FakeLoaded(), FakeImg2Img)
    assert info["ok"] and FakeImg2Img.constructed == []


def _need_haar():
    cv2 = pytest.importorskip("cv2")
    import os
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(path):
        pytest.skip("this OpenCV build ships no Haar cascade files (OpenCV 5 builds may not)")


def test_haar_detector_loads_and_runs_on_blank_image():
    _need_haar()
    assert fr.HaarDetector().detect(Image.new("RGB", (256, 384), (128, 128, 128))) == []


def test_load_detector_falls_back_to_haar_when_model_missing(tmp_path):
    _need_haar()
    assert fr.load_detector(tmp_path / "missing.onnx").name == "haar"


def test_no_detector_available_disables_refinement_cleanly(tmp_path, monkeypatch):
    def broken():
        raise fr.RefinerUnavailable("no cascade")
    monkeypatch.setattr(fr, "HaarDetector", broken)
    with pytest.raises(fr.RefinerUnavailable):
        fr.load_detector(tmp_path / "missing.onnx")
