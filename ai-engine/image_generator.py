import argparse
import json
import os
from pathlib import Path

import numpy as np

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

JOBS_FILE = PROJECT_ROOT / "ai-engine" / "scene_jobs.json"
IMAGE_DIR = PROJECT_ROOT / "storage" / "images"

# A photorealistic SD1.5 checkpoint gives far better faces/sarees than
# the base SD1.5 model. Override with:  set IMAGE_MODEL_ID=<hf repo>
MODEL_ID = os.environ.get(
    "IMAGE_MODEL_ID",
    "SG161222/Realistic_Vision_V5.1_noVAE"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "storage"
    / "models"
    / ("openvino-" + MODEL_ID.split("/")[-1].lower())
)

# SD1.5 is trained near 512px. 448x800 causes duplicated people and
# bad faces. Generate 512x768, then center-crop to 9:16 (432x768).
WIDTH = 512
HEIGHT = 768
CROP_SIZE = (432, 768)

INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.0
BASE_SEED = 1234  # same seed base keeps faces/lighting more consistent

# ==========================================
# STYLE ANCHORS (matched to the reference photo)
# ==========================================

QUALITY = (
    "RAW photo, photorealistic, 85mm lens, shallow depth of field, "
    "soft natural light, detailed skin texture, sharp focus"
)

WOMAN = (
    "beautiful young Indian woman, fair warm skin, long wavy black hair, "
    "small black bindi, gentle smile, light pastel blue sheer chiffon saree "
    "with delicate lace border, gold choker necklace"
)

MAN = (
    "handsome young Indian man, short black hair, "
    "white shirt, dark trousers"
)

# Per-scene composition. Subject text goes FIRST in the prompt because
# CLIP only reads the first 77 tokens.
SCENE_SUBJECTS = {
    1: (
        "portrait of {woman}, facing camera, looking at viewer, "
        "walking on a wet evening street in light rain, "
        "{man} blurred far in background, "
        "three-quarter shot, wet road reflections"
    ),
    2: (
        "{woman} and {man} standing close under one umbrella, "
        "looking into each other's eyes, rainy evening, medium shot"
    ),
    3: (
        "{man} holding {woman}'s hand, walking together in light rain, "
        "both smiling, evening street lights, medium shot"
    ),
    4: (
        "{woman} leaning toward {man}, his arm around her, "
        "tender romantic moment, close-up, rainy evening bokeh lights"
    ),
}

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, 3d render, cgi, "
    "blurry, low resolution, pixelated, plastic skin, "
    "deformed face, asymmetrical eyes, bad anatomy, "
    "extra fingers, missing fingers, extra limbs, "
    "duplicate people, distorted hands, text, watermark, logo, "
    "western clothing, nudity, cleavage, "
    "back view, from behind, facing away, rear view, "
    "cropped head, no face, hair covering face"
)


# ==========================================
# LOAD SCENE JOBS
# ==========================================

def load_scene_jobs():

    if not JOBS_FILE.exists():
        raise FileNotFoundError(
            f"Scene jobs file not found: {JOBS_FILE}"
        )

    with open(JOBS_FILE, "r", encoding="utf-8") as file:
        jobs = json.load(file)

    if not isinstance(jobs, list) or not jobs:
        raise ValueError("No scene jobs found.")

    print(f"Loaded {len(jobs)} scene jobs.")

    return jobs


# ==========================================
# CREATE IMAGE PROMPT (now uses the job)
# ==========================================

def build_prompt(job):

    scene_number = int(job["scene_number"])

    template = SCENE_SUBJECTS.get(scene_number)

    if template:
        subject = template.format(woman=WOMAN, man=MAN)
    else:
        # Fallback for any custom scene: use the story's own prompt.
        subject = job["image_prompt"]

    return f"{subject}, {QUALITY}"


# ==========================================
# LOAD OPENVINO MODEL
# ==========================================

def load_image_pipeline(device):

    try:
        from optimum.intel import OVStableDiffusionPipeline

    except ImportError as error:
        raise RuntimeError(
            "Could not import OVStableDiffusionPipeline. "
            "Check your Optimum Intel installation."
        ) from error

    print("\nLoading OpenVINO Stable Diffusion...")

    if (MODEL_DIR / "model_index.json").exists():

        print("Loading existing converted model...")

        pipeline = OVStableDiffusionPipeline.from_pretrained(
            str(MODEL_DIR),
            device=device,
            compile=False
        )

    else:

        print(f"First-time download/conversion of {MODEL_ID}.")
        print("This requires an internet connection and disk space.")

        pipeline = OVStableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            export=True,
            device=device,
            compile=False
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        pipeline.save_pretrained(str(MODEL_DIR))

        print("Model conversion completed.")

    print("Configuring image dimensions...")

    pipeline.reshape(
        batch_size=1,
        height=HEIGHT,
        width=WIDTH,
        num_images_per_prompt=1
    )

    print("Compiling model...")

    pipeline.compile()

    return pipeline


# ==========================================
# GENERATE ONE IMAGE
# ==========================================

def center_crop(image, size):

    target_w, target_h = size
    w, h = image.size

    left = (w - target_w) // 2
    top = (h - target_h) // 2

    return image.crop((left, top, left + target_w, top + target_h))


def generate_scene_image(pipeline, job):

    scene_number = int(job["scene_number"])

    output_file = IMAGE_DIR / f"scene_{scene_number:02d}.png"

    prompt = build_prompt(job)

    # CLIP limit is 77 tokens: truncate safely.
    tokens = pipeline.tokenizer(
        prompt,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )

    prompt = pipeline.tokenizer.decode(
        tokens.input_ids[0],
        skip_special_tokens=True
    )

    print(f"\nGenerating scene {scene_number}...")
    print("Prompt:", prompt)

    result = pipeline(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=np.random.RandomState(BASE_SEED + scene_number)
    )

    image = center_crop(result.images[0], CROP_SIZE)

    image.save(output_file)

    print(f"Image saved: {output_file}")

    return output_file


# ==========================================
# MAIN IMAGE GENERATION PIPELINE
# ==========================================

def generate_images(device="CPU", force=False, scene_number=None):

    print("\n====================================")
    print(" HEYFUNVERSE AI IMAGE GENERATOR")
    print("====================================")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    jobs = load_scene_jobs()

    if scene_number is not None:
        jobs = [
            job for job in jobs
            if int(job["scene_number"]) == scene_number
        ]

        if not jobs:
            raise ValueError(f"Scene {scene_number} not found.")

    print(f"Scenes selected for generation: {len(jobs)}")

    pipeline = load_image_pipeline(device)

    successful = 0
    skipped = 0

    for job in jobs:

        number = int(job["scene_number"])

        output_file = IMAGE_DIR / f"scene_{number:02d}.png"

        if output_file.exists() and not force:

            print(f"\nScene {number}: Image already exists. Skipping.")

            skipped += 1
            continue

        generate_scene_image(pipeline, job)

        successful += 1

    print("\n====================================")
    print(" IMAGE GENERATION SUMMARY")
    print("====================================")

    print("Total scenes:", len(jobs))
    print("Generated:", successful)
    print("Skipped:", skipped)
    print("Output directory:", IMAGE_DIR)


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="HeyFunverse AI Image Generator"
    )

    parser.add_argument(
        "--scene",
        type=int,
        default=None,
        help="Generate only the specified scene number"
    )

    parser.add_argument(
        "--device",
        choices=["CPU", "GPU", "AUTO"],
        default="CPU"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing scene images"
    )

    args = parser.parse_args()

    generate_images(
        device=args.device,
        force=args.force,
        scene_number=args.scene
    )