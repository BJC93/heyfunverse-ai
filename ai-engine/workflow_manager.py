
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

    parser.add_argument(
        "--theme",
        default="A shy couple meets during a rainy evening in Hyderabad.",
        help="Story theme used with --generate"
    )

    args = parser.parse_args()

    if args.generate:

        import config
        import safe_io
        from story_planner import generate_story

        story = generate_story(theme=args.theme)

        # An existing story is backed up to storage/backups/ before replacement.
        saved = safe_io.write_json_safely(config.STORY_FILE, story)

        if saved["backup"]:
            print(f"Previous story backed up: {saved['backup']}")

    run_pipeline()
