import argparse
import json
import os
import re
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

INFERENCE_STEPS = 35
GUIDANCE_SCALE = 7.0
BASE_SEED = 1234  # same seed base keeps faces/lighting more consistent

# ==========================================
# STYLE ANCHORS (matched to the reference photo)
# ==========================================

QUALITY = (
    "RAW photo, photorealistic, 85mm, detailed symmetrical face, sharp eyes"
)

# Short visual anchors added on top of the descriptions in story.json.
# Keys are the character "id" values from story.json.
LOOK_ANCHORS = {
    "woman": "Indian woman, long wavy black hair, bindi, gold necklace, pastel blue saree, full blouse",
    "man": "Indian man, short black hair, white shirt",
}

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

def first_clause(text):
    """'Wide shot of rain, then close-up' -> 'Wide shot of rain'"""
    return text.split(", then")[0].split(".")[0].strip()


def build_prompt(job):
    """Build the prompt from the scene fields that come from story.json.

    CLIP reads only 77 tokens, so the order is by importance:
    scene -> who is in it -> camera -> lighting -> quality.
    """

    scene_text = job["image_prompt"].strip().rstrip(".")

    # Anchor only the characters that actually appear in this scene.
    haystack = (job["image_prompt"] + " " + job["description"]).lower()

    anchors = []
    for character in job.get("characters", []):
        cid = character["id"]
        if re.search(rf"\b{re.escape(cid.lower())}\b", haystack):
            anchors.append(
                LOOK_ANCHORS.get(cid, character["description"])
            )

    # Quality/face words come before camera+lighting so that, if the
    # 77-token limit truncates anything, it is the least important part.
    parts = [
        scene_text,
        "; ".join(anchors),
        QUALITY,
        first_clause(job["camera"]),
        job["lighting"].rstrip("."),
    ]

    return ", ".join(p for p in parts if p)


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

# ==========================================
# FACE REFINEMENT (second pass on each detected face)
# ==========================================

FACE_PROMPT = (
    "detailed face of an Indian person, symmetrical eyes, natural skin, "
    "sharp focus, soft smile, RAW photo"
)
FACE_NEGATIVE = (
    "deformed face, distorted eyes, crossed eyes, uneven eyes, "
    "bad teeth, malformed mouth, blurry, cartoon, plastic skin, "
    "extra face, text, watermark"
)
FACE_STRENGTH = 0.35   # low = keep identity, only clean up details
FACE_SIZE = 512


def load_face_refiner(device):

    try:
        from optimum.intel import OVStableDiffusionImg2ImgPipeline
        import cv2  # noqa: F401
    except ImportError as error:
        print(f"Face refinement disabled ({error}).")
        return None

    print("\nLoading face refiner (img2img)...")

    refiner = OVStableDiffusionImg2ImgPipeline.from_pretrained(
        str(MODEL_DIR),
        device=device,
        compile=False
    )

    refiner.reshape(
        batch_size=1,
        height=FACE_SIZE,
        width=FACE_SIZE,
        num_images_per_prompt=1
    )

    refiner.compile()

    return refiner


def detect_faces(image):

    import cv2

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    boxes = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )

    # biggest faces first, max 2 people
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:2]

    return [tuple(int(v) for v in b) for b in boxes]


def refine_faces(refiner, image, seed):

    from PIL import Image, ImageDraw, ImageFilter

    if refiner is None:
        return image

    faces = detect_faces(image)

    print(f"Faces detected: {len(faces)}")

    for index, (x, y, w, h) in enumerate(faces):

        # square crop with generous padding (hair, jaw, forehead)
        side = int(max(w, h) * 1.8)
        cx, cy = x + w // 2, y + h // 2

        left = max(0, min(image.width - side, cx - side // 2))
        top = max(0, min(image.height - side, cy - side // 2))
        side = min(side, image.width, image.height)

        box = (left, top, left + side, top + side)

        crop = image.crop(box).resize(
            (FACE_SIZE, FACE_SIZE), Image.LANCZOS
        )

        fixed = refiner(
            prompt=FACE_PROMPT,
            negative_prompt=FACE_NEGATIVE,
            image=crop,
            strength=FACE_STRENGTH,
            num_inference_steps=30,
            guidance_scale=6.5,
            generator=np.random.RandomState(seed + index)
        ).images[0].resize((side, side), Image.LANCZOS)

        # feathered oval mask so the patch blends in
        mask = Image.new("L", (side, side), 0)
        pad = int(side * 0.12)
        ImageDraw.Draw(mask).ellipse(
            (pad, pad, side - pad, side - pad), fill=255
        )
        mask = mask.filter(ImageFilter.GaussianBlur(side * 0.08))

        image.paste(fixed, box[:2], mask)

    return image



def center_crop(image, size):

    target_w, target_h = size
    w, h = image.size

    left = (w - target_w) // 2
    top = (h - target_h) // 2

    return image.crop((left, top, left + target_w, top + target_h))


def generate_scene_image(pipeline, job, refiner=None):

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

    image = refine_faces(
        refiner, result.images[0], BASE_SEED + scene_number
    )

    image = center_crop(image, CROP_SIZE)

    image.save(output_file)

    print(f"Image saved: {output_file}")

    return output_file


# ==========================================
# MAIN IMAGE GENERATION PIPELINE
# ==========================================

def generate_images(device="CPU", force=False, scene_number=None, refine=True):

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

    refiner = load_face_refiner(device) if refine else None

    successful = 0
    skipped = 0

    for job in jobs:

        number = int(job["scene_number"])

        output_file = IMAGE_DIR / f"scene_{number:02d}.png"

        if output_file.exists() and not force:

            print(f"\nScene {number}: Image already exists. Skipping.")

            skipped += 1
            continue

        generate_scene_image(pipeline, job, refiner)

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

    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip the face refinement pass (faster)"
    )

    args = parser.parse_args()

    generate_images(
        device=args.device,
        force=args.force,
        scene_number=args.scene,
        refine=not args.no_refine
    )