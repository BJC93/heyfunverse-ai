"""Optional face refinement that never loads a second model copy.

How it works
------------
1. Detect faces in the generated 512x768 image.
2. For each face, crop a portrait-shaped region (same 512x768 shape as the
   main image, so the already compiled static-shape OpenVINO models fit).
3. Run a low-strength img2img pass on that crop and blend it back with a
   feathered mask.

The img2img pipeline is assembled from the components of the pipeline that
is ALREADY loaded (see :func:`build_shared_img2img`). This module never calls
``from_pretrained``. If the components cannot be shared safely, refinement is
simply unavailable and the caller carries on without it.

Detectors: YuNet (small ONNX file, optional) or OpenCV's built-in Haar
cascade as a fallback. OpenCV is imported lazily.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class RefinerUnavailable(RuntimeError):
    """Face refinement cannot run safely in this setup."""


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    @property
    def area(self):
        return self.w * self.h


# ------------------------------------------------------------------
# Detectors
# ------------------------------------------------------------------

class HaarDetector:
    """OpenCV's bundled frontal-face Haar cascade (no download needed)."""

    name = "haar"

    def __init__(self):
        import cv2

        self._cv2 = cv2
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)

        if self._cascade.empty():
            raise RefinerUnavailable("Could not load the Haar cascade file.")

    def detect(self, image):
        cv2 = self._cv2
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

        boxes = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )

        return [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]


class YuNetDetector:
    """YuNet via cv2.FaceDetectorYN (needs the small ONNX model file)."""

    name = "yunet"

    def __init__(self, model_path, score_threshold=0.7, nms_threshold=0.3, top_k=50):
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RefinerUnavailable("This OpenCV build has no FaceDetectorYN.")

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320),
            score_threshold, nms_threshold, top_k
        )

    def detect(self, image):
        cv2 = self._cv2
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]

        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(bgr)

        if faces is None:
            return []

        # Row layout: x, y, w, h, 10 landmark values, score.
        return [
            FaceBox(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[14]))
            for f in faces
        ]


def load_detector(model_file=None):
    """YuNet if its model file exists and loads, otherwise Haar."""

    if model_file is None:
        import config

        model_file = config.YUNET_MODEL_FILE

    model_file = Path(model_file)

    if model_file.is_file():
        try:
            return YuNetDetector(model_file)
        except Exception as error:  # fall back instead of failing the run
            print(f"YuNet unavailable ({error}); using Haar cascade.")

    return HaarDetector()


# ------------------------------------------------------------------
# Geometry and blending
# ------------------------------------------------------------------

def crop_box_for_face(face, image_size, out_size, scale=3.2):
    """Portrait crop (left, top, right, bottom) around ``face``.

    The box has the same aspect ratio as ``out_size`` so it can be resized to
    exactly ``out_size`` without distortion, and always lies inside the image.
    """

    img_w, img_h = image_size
    out_w, out_h = out_size
    ratio = out_w / out_h

    box_h = min(float(img_h), face.h * scale)
    box_w = box_h * ratio

    if box_w > img_w:
        box_w = float(img_w)
        box_h = box_w / ratio

    box_w = max(1, int(round(box_w)))
    box_h = max(1, int(round(box_h)))

    center_x = face.x + face.w / 2
    # Put the face slightly above centre so neck and shoulders are included.
    center_y = face.y + face.h / 2 + 0.15 * box_h

    left = int(round(center_x - box_w / 2))
    top = int(round(center_y - box_h / 2))

    left = max(0, min(left, img_w - box_w))
    top = max(0, min(top, img_h - box_h))

    return left, top, left + box_w, top + box_h


def feather_mask(size, margin=0.12, blur=0.08):
    """Soft oval mask (white in the middle) used to blend the refined crop."""

    from PIL import Image, ImageDraw, ImageFilter

    width, height = size
    mask = Image.new("L", size, 0)

    pad_x = int(width * margin)
    pad_y = int(height * margin)

    ImageDraw.Draw(mask).ellipse(
        (pad_x, pad_y, width - pad_x, height - pad_y), fill=255
    )

    return mask.filter(ImageFilter.GaussianBlur(min(width, height) * blur))


# ------------------------------------------------------------------
# Refinement
# ------------------------------------------------------------------

def refine_faces(
    image,
    detector,
    img2img,
    *,
    prompt,
    negative_prompt,
    out_size,
    strength,
    steps,
    seed,
    guidance_scale=6.5,
    max_faces=2,
    min_face=40
):
    """Refine up to ``max_faces`` faces. Returns (new_image, report).

    ``img2img`` is any callable accepting the keyword arguments prompt,
    negative_prompt, image, strength, num_inference_steps, guidance_scale and
    generator, and returning a PIL image of the same size as ``image``.
    """

    from PIL import Image

    result = image.copy()
    report = []

    faces = [f for f in detector.detect(result) if min(f.w, f.h) >= min_face]
    faces.sort(key=lambda f: f.area, reverse=True)

    for index, face in enumerate(faces[:max_faces]):

        box = crop_box_for_face(face, result.size, out_size)
        left, top, right, bottom = box
        crop_size = (right - left, bottom - top)

        crop = result.crop(box).resize(out_size, Image.LANCZOS)

        started = time.perf_counter()

        fixed = img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=crop,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=np.random.RandomState(seed + index)
        )

        fixed = fixed.resize(crop_size, Image.LANCZOS)

        result.paste(fixed, (left, top), feather_mask(crop_size))

        report.append({
            "face": [face.x, face.y, face.w, face.h],
            "crop": list(box),
            "detector": getattr(detector, "name", "unknown"),
            "seconds": round(time.perf_counter() - started, 2),
        })

    return result, report


# ------------------------------------------------------------------
# Sharing the already loaded pipeline (never from_pretrained)
# ------------------------------------------------------------------

def map_components(cls, source):
    """Match the constructor parameters of ``cls`` to attributes of the loaded
    pipeline ``source``. Raises :class:`RefinerUnavailable` if a required
    parameter has no match or a needed component is missing."""

    signature = inspect.signature(cls.__init__)

    kwargs = {}
    missing = []

    for name, parameter in signature.parameters.items():

        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD
        ):
            continue

        if hasattr(source, name):
            kwargs[name] = getattr(source, name)

        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)

    if missing:
        raise RefinerUnavailable(
            "The loaded pipeline does not expose these components: "
            + ", ".join(missing)
        )

    for needed in ("unet", "vae_encoder"):
        if needed in kwargs and kwargs[needed] is None:
            raise RefinerUnavailable(f"The loaded pipeline has no {needed}.")

    return kwargs


def describe_sharing(source, cls=None):
    """Read-only report for check_env: can the img2img pipeline be assembled
    from ``source``? Constructs nothing."""

    if cls is None:
        from optimum.intel import OVStableDiffusionImg2ImgPipeline as cls

    signature = inspect.signature(cls.__init__)

    try:
        kwargs = map_components(cls, source)
    except RefinerUnavailable as error:
        return {"ok": False, "signature": str(signature), "detail": str(error)}

    return {
        "ok": True,
        "signature": str(signature),
        "detail": "parameters matched: " + ", ".join(sorted(kwargs)),
    }


def build_shared_img2img(pipeline, img2img_cls=None):
    """Assemble an img2img pipeline from ``pipeline``'s existing components.

    No model is read from disk and nothing is recompiled, so memory does not
    double. Raises :class:`RefinerUnavailable` when that is not possible.
    """

    if img2img_cls is None:
        try:
            from optimum.intel import OVStableDiffusionImg2ImgPipeline as img2img_cls
        except ImportError as error:
            raise RefinerUnavailable(
                "optimum-intel is not installed: " + str(error)
            ) from error

    kwargs = map_components(img2img_cls, pipeline)

    try:
        return img2img_cls(**kwargs)
    except Exception as error:
        raise RefinerUnavailable(
            f"Could not assemble the img2img pipeline from loaded parts: {error}"
        ) from error
