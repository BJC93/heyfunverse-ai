
import argparse
import json
from pathlib import Path

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

JOBS_FILE = PROJECT_ROOT / "ai-engine" / "scene_jobs.json"
IMAGE_DIR = PROJECT_ROOT / "storage" / "images"

MODEL_DIR = (
    PROJECT_ROOT
    / "storage"
    / "models"
    / "openvino-sd15"
)

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

WIDTH = 448
HEIGHT = 800

INFERENCE_STEPS = 25
GUIDANCE_SCALE = 6.5

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, 3d render, "
    "blurry, low resolution, pixelated, unrealistic face, "
    "deformed face, asymmetrical eyes, bad anatomy, "
    "extra fingers, missing fingers, extra limbs, "
    "duplicate people, distorted hands, text, watermark, logo"
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
# CREATE IMAGE PROMPT
# ==========================================

def build_prompt(job):
    prompt = (
        "Photorealistic cinematic Indian romantic drama, "
        "realistic human faces, natural skin texture, "
        "professional movie cinematography. "
        "A young Indian woman with long black hair, "
        "wearing an elegant pastel blue saree, "
        "walking alone on a rainy street at evening. "
        "A young Indian man wearing a clean white shirt "
        "and dark trousers is visible in the background, "
        "emerging from behind a building. "
        "Natural body proportions, realistic clothing, "
        "beautiful rainy atmosphere, cinematic composition, "
        "soft reflections on wet road, detailed background."
    )

    return prompt


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

        print("First-time model download and conversion.")
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

def generate_scene_image(pipeline, job):

    scene_number = int(job["scene_number"])

    output_file = IMAGE_DIR / f"scene_{scene_number:02d}.png"

    prompt = build_prompt(job)
    
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

    print(
        "Prompt tokens:",
        len(tokens.input_ids[0])
    )

    print(f"\nGenerating scene {scene_number}...")

    result = pipeline(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=1
    )

    image = result.images[0]

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

        scene_number = int(job["scene_number"])

        output_file = IMAGE_DIR / f"scene_{scene_number:02d}.png"

        if output_file.exists() and not force:

            print(
                f"\nScene {scene_number}: Image already exists. Skipping."
            )

            skipped += 1
            continue

        try:

            generate_scene_image(pipeline, job)

            successful += 1

        except Exception as error:

            print(
                f"\nScene {scene_number} failed: {error}"
            )

            raise

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
