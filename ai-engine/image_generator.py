"""Scene image generator (OpenVINO + Stable Diffusion 1.5 family).

Prompts are built from the scene data in scene_jobs.json (which comes from
story.json), not from hardcoded text. Existing images are never overwritten
silently: --force backs the old image up first.

Examples:
    python ai-engine/image_generator.py --dry-run          # show prompts only
    python ai-engine/image_generator.py --scene 1 --force
    python ai-engine/image_generator.py --device AUTO
    python ai-engine/image_generator.py --refine-faces     # optional, see docs
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np

import config
import safe_io


class ModelNotInstalledError(RuntimeError):
    """The converted OpenVINO model is not on disk and downloads are disabled."""


# ==========================================
# LOAD SCENE JOBS
# ==========================================

def load_scene_jobs(jobs_file=None):

    jobs_file = Path(jobs_file or config.JOBS_FILE)

    if not jobs_file.exists():
        raise FileNotFoundError(
            f"Scene jobs file not found: {jobs_file}"
        )

    with open(jobs_file, "r", encoding="utf-8") as file:
        jobs = json.load(file)

    if not isinstance(jobs, list) or not jobs:
        raise ValueError("No scene jobs found.")

    print(f"Loaded {len(jobs)} scene jobs.")

    return jobs


# ==========================================
# PROMPT BUILDING (pure functions, no model needed)
# ==========================================

def approx_tokens(text):
    """Rough CLIP token estimate used when no tokenizer is loaded (dry runs).
    Slightly pessimistic on purpose."""

    pieces = re.findall(r"\w+|[^\w\s]", text)

    return math.ceil(len(pieces) * 1.25)


def first_clause(text):
    """'Wide shot of rain, then close-up' -> 'Wide shot of rain'."""

    return text.split(", then")[0].split(".")[0].strip()


def character_look(character):
    """Visual description of one character, most specific source first."""

    identity = (character.get("visual_identity") or "").strip()

    if identity:
        return identity

    legacy = config.LEGACY_LOOK_ANCHORS.get(character.get("id", ""))

    if legacy:
        return legacy

    return (character.get("description") or "").strip()


def characters_in_scene(job):
    """Characters whose id appears as a whole word in the scene text."""

    haystack = (
        str(job.get("image_prompt", "")) + " " + str(job.get("description", ""))
    ).lower()

    found = []

    for character in job.get("characters", []):

        cid = str(character.get("id", "")).lower()

        if cid and re.search(rf"\b{re.escape(cid)}\b", haystack):
            found.append(character)

    return found


def build_prompt_parts(job):
    """Prompt pieces in priority order: [(name, text), ...].

    CLIP reads only 77 tokens, so the most important parts come first and the
    least important are dropped when the budget runs out.
    """

    parts = [
        ("scene", str(job["image_prompt"]).strip().rstrip(".")),
        # Quality words come before the character descriptions: the scene
        # text already names clothing, and these words keep faces sharp.
        ("quality", config.QUALITY_SUFFIX),
    ]

    for character in characters_in_scene(job):
        parts.append((f"character:{character['id']}", character_look(character)))

    parts.append(("camera", first_clause(str(job.get("camera", "")))))
    parts.append(("lighting", str(job.get("lighting", "")).strip().rstrip(".")))

    return parts


def fit_prompt(parts, count_tokens, max_tokens):
    """Join as many parts as fit in ``max_tokens``. The first part is always
    kept. Returns (prompt, names_of_dropped_parts)."""

    chosen = []
    dropped = []
    used = 0

    for name, text in parts:

        text = text.strip().strip(",").strip()

        if not text:
            continue

        cost = count_tokens(text) + (1 if chosen else 0)

        if not chosen or used + cost <= max_tokens:
            chosen.append(text)
            used += cost
        else:
            dropped.append(name)

    return ", ".join(chosen), dropped


def build_prompt_with_report(job, count_tokens=None, max_tokens=None):

    count_tokens = count_tokens or approx_tokens
    max_tokens = max_tokens or config.MAX_PROMPT_TOKENS

    return fit_prompt(build_prompt_parts(job), count_tokens, max_tokens)


def build_prompt(job, count_tokens=None):

    prompt, _ = build_prompt_with_report(job, count_tokens)

    return prompt


def _tokenizer_counter(pipeline):

    tokenizer = getattr(pipeline, "tokenizer", None)

    if tokenizer is None:
        return approx_tokens

    def count(text):
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    return count


def _truncate_to_clip(pipeline, prompt):
    """Safety net: cut at 77 tokens if anything slipped through."""

    tokenizer = getattr(pipeline, "tokenizer", None)

    if tokenizer is None:
        return prompt

    tokens = tokenizer(
        prompt, truncation=True, max_length=77, return_tensors="pt"
    )

    return tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)


# ==========================================
# LOAD OPENVINO MODEL
# ==========================================

def load_image_pipeline(device=None, allow_download=True):

    device = device or config.DEFAULT_DEVICE

    model_ready = (config.MODEL_DIR / "model_index.json").exists()

    if not model_ready and not allow_download:
        raise ModelNotInstalledError(
            f"Converted model not found in {config.MODEL_DIR}. "
            "Run image_generator.py once to download and convert it."
        )

    try:
        from optimum.intel import OVStableDiffusionPipeline

    except ImportError as error:
        raise RuntimeError(
            "Could not import OVStableDiffusionPipeline. "
            "Check your Optimum Intel installation."
        ) from error

    print(f"\nLoading OpenVINO Stable Diffusion (device: {device})...")

    if model_ready:

        print("Loading existing converted model...")

        pipeline = OVStableDiffusionPipeline.from_pretrained(
            str(config.MODEL_DIR),
            device=device,
            compile=False
        )

    else:

        print(f"First-time download/conversion of {config.IMAGE_MODEL_ID}.")
        print("This requires an internet connection and disk space.")

        pipeline = OVStableDiffusionPipeline.from_pretrained(
            config.IMAGE_MODEL_ID,
            export=True,
            device=device,
            compile=False
        )

        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        pipeline.save_pretrained(str(config.MODEL_DIR))

        print("Model conversion completed.")

    print("Configuring image dimensions...")

    pipeline.reshape(
        batch_size=1,
        height=config.HEIGHT,
        width=config.WIDTH,
        num_images_per_prompt=1
    )

    print("Compiling model...")

    pipeline.compile()

    return pipeline


def make_face_refiner(pipeline):
    """Face refinement that reuses the loaded pipeline. Returns a function
    ``refine(image, seed) -> (image, report)`` or None if it cannot run."""

    import face_refine

    try:
        shared = face_refine.build_shared_img2img(pipeline)
        detector = face_refine.load_detector()

    except face_refine.RefinerUnavailable as error:
        print(f"Face refinement disabled: {error}")
        return None

    print(f"Face refinement enabled (detector: {detector.name}).")

    def img2img(**kwargs):
        return shared(**kwargs).images[0]

    def refine(image, seed):
        return face_refine.refine_faces(
            image,
            detector,
            img2img,
            prompt=config.FACE_PROMPT,
            negative_prompt=config.FACE_NEGATIVE_PROMPT,
            out_size=(config.WIDTH, config.HEIGHT),
            strength=config.FACE_REFINE_STRENGTH,
            steps=config.FACE_REFINE_STEPS,
            seed=seed,
            guidance_scale=config.FACE_REFINE_GUIDANCE,
            max_faces=config.FACE_MAX_FACES,
            min_face=config.FACE_MIN_SIZE_PX
        )

    return refine


# ==========================================
# GENERATE ONE IMAGE
# ==========================================

def center_crop(image, size):

    target_w, target_h = size
    w, h = image.size

    left = (w - target_w) // 2
    top = (h - target_h) // 2

    return image.crop((left, top, left + target_w, top + target_h))


def generate_scene_image(pipeline, job, refiner=None, image_dir=None):

    scene_number = int(job["scene_number"])

    image_dir = Path(image_dir or config.IMAGE_DIR)
    output_file = image_dir / f"scene_{scene_number:02d}.png"
    meta_file = image_dir / f"scene_{scene_number:02d}.json"

    prompt, dropped = build_prompt_with_report(
        job, _tokenizer_counter(pipeline)
    )
    prompt = _truncate_to_clip(pipeline, prompt)

    seed = config.BASE_SEED + scene_number

    print(f"\nGenerating scene {scene_number}...")
    print("Prompt:", prompt)

    if dropped:
        print("Dropped to fit 77 tokens:", ", ".join(dropped))

    started = time.perf_counter()

    result = pipeline(
        prompt=prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        width=config.WIDTH,
        height=config.HEIGHT,
        num_inference_steps=config.INFERENCE_STEPS,
        guidance_scale=config.GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=np.random.RandomState(seed)
    )

    image = result.images[0]
    refine_report = None

    if refiner is not None:
        image, refine_report = refiner(image, seed)

    image = center_crop(image, config.CROP_SIZE)

    image_dir.mkdir(parents=True, exist_ok=True)

    backup = safe_io.backup_file(output_file)

    if backup:
        print(f"Previous image backed up: {backup}")

    image.save(output_file)

    safe_io.write_json_safely(meta_file, {
        "scene_number": scene_number,
        "prompt": prompt,
        "dropped_parts": dropped,
        "seed": seed,
        "steps": config.INFERENCE_STEPS,
        "guidance_scale": config.GUIDANCE_SCALE,
        "size": list(config.CROP_SIZE),
        "model": config.IMAGE_MODEL_ID,
        "face_refinement": refine_report,
        "seconds": round(time.perf_counter() - started, 2),
    })

    print(f"Image saved: {output_file}")

    return output_file


# ==========================================
# MAIN IMAGE GENERATION PIPELINE
# ==========================================

def select_jobs(jobs, scene_number=None):

    if scene_number is None:
        return jobs

    selected = [j for j in jobs if int(j["scene_number"]) == scene_number]

    if not selected:
        raise ValueError(f"Scene {scene_number} not found.")

    return selected


def generate_images(
    device=None,
    force=False,
    scene_number=None,
    refine_faces=False,
    dry_run=False
):

    print("\n====================================")
    print(" HEYFUNVERSE AI IMAGE GENERATOR")
    print("====================================")

    jobs = select_jobs(load_scene_jobs(), scene_number)

    print(f"Scenes selected: {len(jobs)}")

    if dry_run:

        print("\nDRY RUN: no model is loaded and no files are written.\n")

        for job in jobs:

            prompt, dropped = build_prompt_with_report(job)

            print(
                f"Scene {job['scene_number']} "
                f"(~{approx_tokens(prompt)} tokens)"
            )
            print("  ", prompt)

            if dropped:
                print("   dropped:", ", ".join(dropped))

        return

    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = load_image_pipeline(device)

    refiner = make_face_refiner(pipeline) if refine_faces else None

    successful = 0
    skipped = 0

    for job in jobs:

        number = int(job["scene_number"])

        output_file = config.IMAGE_DIR / f"scene_{number:02d}.png"

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
    print("Output directory:", config.IMAGE_DIR)


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

def build_parser():

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
        choices=list(config.VALID_DEVICES),
        default=config.DEFAULT_DEVICE,
        help="OpenVINO device (default: %(default)s). "
             "GPU/AUTO are not assumed to work: run benchmark.py first."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing images (old ones are backed up first)"
    )

    parser.add_argument(
        "--refine-faces",
        action="store_true",
        help="Optional face refinement that reuses the loaded model"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompts and exit without loading any model"
    )

    return parser


if __name__ == "__main__":

    args = build_parser().parse_args()

    generate_images(
        device=args.device,
        force=args.force,
        scene_number=args.scene,
        refine_faces=args.refine_faces,
        dry_run=args.dry_run
    )
