
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

STORY_FILE = PROJECT_ROOT / "storage" / "story.json"
JOBS_FILE = BASE_DIR / "scene_jobs.json"


def load_scene_plan(file_path=STORY_FILE):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Story JSON not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        plan = json.load(file)

    validate_scene_plan(plan)

    return plan


def validate_scene_plan(plan):

    required_fields = [
        "title",
        "style",
        "characters",
        "scenes"
    ]

    if not isinstance(plan, dict):
        raise ValueError("Story must be a JSON object")

    for field in required_fields:
        if field not in plan:
            raise ValueError(f"Missing field: {field}")

    if not isinstance(plan["characters"], list) or not plan["characters"]:
        raise ValueError("Characters must be a non-empty list")

    character_ids = set()

    for character in plan["characters"]:

        if not isinstance(character, dict):
            raise ValueError("Invalid character object")

        if not character.get("id") or not character.get("description"):
            raise ValueError("Character ID and description are required")

        if character["id"] in character_ids:
            raise ValueError("Duplicate character ID")

        character_ids.add(character["id"])

    scenes = plan["scenes"]

    if not isinstance(scenes, list) or len(scenes) != 4:
        raise ValueError("Exactly 4 scenes are required")

    required_scene_fields = [
        "scene_number",
        "start",
        "end",
        "description",
        "image_prompt",
        "motion_prompt",
        "camera",
        "lighting"
    ]

    for index, scene in enumerate(scenes):

        if not isinstance(scene, dict):
            raise ValueError(f"Scene {index + 1} must be an object")

        for field in required_scene_fields:

            if field not in scene:
                raise ValueError(
                    f"Scene {index + 1} missing {field}"
                )

        if not isinstance(scene["start"], (int, float)):
            raise ValueError("Scene start must be numeric")

        if not isinstance(scene["end"], (int, float)):
            raise ValueError("Scene end must be numeric")

        expected_start = index * 5
        expected_end = expected_start + 5

        if scene["start"] != expected_start:
            raise ValueError(
                f"Scene {index + 1} must start at {expected_start}"
            )

        if scene["end"] != expected_end:
            raise ValueError(
                f"Scene {index + 1} must end at {expected_end}"
            )

        for field in [
            "description",
            "image_prompt",
            "motion_prompt",
            "camera",
            "lighting"
        ]:
            if not isinstance(scene[field], str) or not scene[field].strip():
                raise ValueError(
                    f"Scene {index + 1}: invalid {field}"
                )

        scene["scene_number"] = index + 1

    print("Scene plan validation successful!")


def prepare_scene_jobs(plan):

    jobs = []

    for scene in plan["scenes"]:

        character_prompt = "\n".join(
            f"{character['id']}: "
            f"{character.get('visual_identity', character['description'])}"
            for character in plan["characters"]
        )

        jobs.append({

            "project_title": plan["title"],

            "scene_number": scene["scene_number"],

            "start": scene["start"],

            "end": scene["end"],

            "duration": scene["end"] - scene["start"],

            "style": plan["style"],

            "character_prompt": character_prompt,

            "characters": plan["characters"],

            "description": scene["description"],

            "image_prompt": scene["image_prompt"],

            "motion_prompt": scene["motion_prompt"],

            "camera": scene["camera"],

            "lighting": scene["lighting"],

            "status": "pending"

        })

    return jobs


def save_scene_jobs(jobs):

    JOBS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(JOBS_FILE, "w", encoding="utf-8") as file:

        json.dump(
            jobs,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Scene jobs saved: {JOBS_FILE}")


if __name__ == "__main__":

    try:

        plan = load_scene_plan()

        jobs = prepare_scene_jobs(plan)

        print(f"\nProject: {plan['title']}")
        print(f"Total scenes: {len(jobs)}")

        for job in jobs:

            print(
                f"Scene {job['scene_number']}: "
                f"{job['duration']} seconds"
            )

        save_scene_jobs(jobs)

    except Exception as error:

        print(f"Scene processing failed: {error}")
        raise
