"""Central configuration for HeyFunverse AI.

Every path and tunable value lives here so that all tools read the same
settings, whether the Python tools are driven by Claude Code or by the local
Qwen fallback. Nothing here imports heavy libraries (no OpenVINO, torch or
OpenCV), so it is safe to import from tests and from check_env.

Override a value with an environment variable (PowerShell example):

    $env:HEYFUN_DEVICE = "AUTO"
    $env:HEYFUN_STEPS  = "20"
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_ENGINE_DIR = PROJECT_ROOT / "ai-engine"


# ------------------------------------------------------------------
# Small helpers for reading environment variables
# ------------------------------------------------------------------

def _env_str(name, default):
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw!r}") from error


def _env_path(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw).expanduser().resolve()


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

STORAGE_DIR = _env_path("HEYFUN_STORAGE_DIR", PROJECT_ROOT / "storage")

STORY_FILE = STORAGE_DIR / "story.json"
JOBS_FILE = AI_ENGINE_DIR / "scene_jobs.json"

IMAGE_DIR = STORAGE_DIR / "images"
MODELS_DIR = STORAGE_DIR / "models"
FACE_MODEL_DIR = MODELS_DIR / "face"
BACKUP_DIR = STORAGE_DIR / "backups"
BENCHMARK_DIR = STORAGE_DIR / "benchmarks"
FONT_DIR = STORAGE_DIR / "assets" / "fonts"


# ------------------------------------------------------------------
# Ollama (local Qwen fallback)
# ------------------------------------------------------------------

OLLAMA_BASE_URL = _env_str("HEYFUN_OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_GENERATE_URL = OLLAMA_BASE_URL + "/api/generate"
OLLAMA_TAGS_URL = OLLAMA_BASE_URL + "/api/tags"
OLLAMA_MODEL = _env_str("HEYFUN_OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SECONDS = _env_int("HEYFUN_OLLAMA_TIMEOUT", 300)


# ------------------------------------------------------------------
# Image generation (OpenVINO + Stable Diffusion 1.5 family)
# ------------------------------------------------------------------

VALID_DEVICES = ("CPU", "GPU", "AUTO")

DEFAULT_DEVICE = _env_str("HEYFUN_DEVICE", "CPU").upper()
if DEFAULT_DEVICE not in VALID_DEVICES:
    raise ValueError(
        f"HEYFUN_DEVICE must be one of {VALID_DEVICES}, got {DEFAULT_DEVICE!r}"
    )

# HEYFUN_IMAGE_MODEL is the preferred name; IMAGE_MODEL_ID is kept because
# earlier versions of image_generator.py read it.
IMAGE_MODEL_ID = (
    os.environ.get("HEYFUN_IMAGE_MODEL")
    or os.environ.get("IMAGE_MODEL_ID")
    or "SG161222/Realistic_Vision_V5.1_noVAE"
)

MODEL_DIR = MODELS_DIR / ("openvino-" + IMAGE_MODEL_ID.split("/")[-1].lower())

# SD 1.5 is trained near 512 px. Generate 512x768 and centre-crop to 9:16.
WIDTH = 512
HEIGHT = 768
CROP_SIZE = (432, 768)

INFERENCE_STEPS = _env_int("HEYFUN_STEPS", 30)
GUIDANCE_SCALE = _env_float("HEYFUN_GUIDANCE", 7.0)
BASE_SEED = _env_int("HEYFUN_SEED", 1234)

# CLIP reads 77 tokens including the start and end tokens.
MAX_PROMPT_TOKENS = 75


# ------------------------------------------------------------------
# Prompt text
# ------------------------------------------------------------------

QUALITY_SUFFIX = (
    "RAW photo, photorealistic, 85mm, detailed face, sharp eyes"
)

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, 3d render, cgi, "
    "blurry, low resolution, pixelated, plastic skin, "
    "deformed face, asymmetrical eyes, bad anatomy, "
    "extra fingers, missing fingers, extra limbs, "
    "duplicate people, distorted hands, text, watermark, logo, "
    "western clothing, nudity, cleavage, "
    "back view, from behind, facing away, rear view, "
    "cropped head, no face, hair covering face, "
    "barefoot, bare feet, no shoes, exposed midriff, bare stomach, "
    "see-through, revealing clothes, "
    "ugly, distorted eyes, crossed eyes, uneven eyes, bad teeth, "
    "malformed mouth, mutated face, disfigured, double face"
)

# Used ONLY when a character in the story has no "visual_identity" field.
# Phase 1 replaces this with a per-story character bible.
LEGACY_LOOK_ANCHORS = {
    "woman": (
        "Indian woman, long wavy black hair, bindi, gold necklace, "
        "pastel blue saree, full blouse"
    ),
    "man": "Indian man, short black hair, white shirt",
}


# ------------------------------------------------------------------
# Face refinement (optional, off by default)
# ------------------------------------------------------------------

FACE_PROMPT = (
    "detailed face of an Indian person, symmetrical eyes, natural skin, "
    "sharp focus, soft smile, RAW photo"
)
FACE_NEGATIVE_PROMPT = (
    "deformed face, distorted eyes, crossed eyes, uneven eyes, "
    "bad teeth, malformed mouth, blurry, cartoon, plastic skin, "
    "extra face, text, watermark"
)
FACE_REFINE_STRENGTH = _env_float("HEYFUN_FACE_STRENGTH", 0.35)
FACE_REFINE_STEPS = _env_int("HEYFUN_FACE_STEPS", 30)
FACE_REFINE_GUIDANCE = 6.5
FACE_MIN_SIZE_PX = 40
FACE_MAX_FACES = 2

# Small optional models (YuNet detector, SFace recogniser). See docs/FACE_MODELS.md.
YUNET_MODEL_FILE = FACE_MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_FILE = FACE_MODEL_DIR / "face_recognition_sface_2021dec.onnx"
