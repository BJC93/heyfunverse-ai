
import argparse

from scene_processor import (
    load_scene_plan,
    prepare_scene_jobs,
    save_scene_jobs
)


def run_pipeline():

    print("\n================================")
    print(" HEYFUNVERSE AI VIDEO GENERATOR")
    print("================================\n")

    print("Step 1: Loading story...")

    story = load_scene_plan()

    print("Story:", story["title"])

    print("\nStep 2: Preparing scenes...")

    jobs = prepare_scene_jobs(story)

    print("\nStep 3: Saving scene jobs...")

    save_scene_jobs(jobs)

    print("\n================================")
    print("STORY PROCESSING COMPLETED")
    print("================================")

    print("Total scenes:", len(jobs))
    print("Total duration:", sum(j["duration"] for j in jobs))
    print("Next: Image generation")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a new story using Ollama"
    )

    args = parser.parse_args()

    if args.generate:

        from story_planner import generate_story
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        output = root / "storage" / "story.json"

        story = generate_story(
            theme="A shy couple meets during a rainy evening in Hyderabad."
        )

        output.parent.mkdir(parents=True, exist_ok=True)

        output.write_text(
            json.dumps(story, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    run_pipeline()
