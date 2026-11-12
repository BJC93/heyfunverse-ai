
import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"


def validate_story(story):
    """Validate and normalize the generated story."""

    if not isinstance(story, dict):
        raise ValueError("Story must be a JSON object")

    scenes = story.get("scenes")

    if not isinstance(scenes, list) or len(scenes) != 4:
        raise ValueError("Story must contain exactly 4 scenes")

    required_fields = [
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

        for field in required_fields:
            if field not in scene:
                raise ValueError(
                    f"Scene {index + 1} missing field: {field}"
                )

            if not isinstance(scene[field], (str, int, float)):
                raise ValueError(
                    f"Scene {index + 1} has invalid field: {field}"
                )

        # Normalize scene numbers and timing.
        scene["scene_number"] = index + 1
        scene["start"] = index * 5
        scene["end"] = (index + 1) * 5

    story.setdefault("title", "చెప్పలేని ప్రేమ")
    story.setdefault("style", "Telugu romantic cinema")

    return story


def generate_story(theme, duration=20):

    if duration != 20:
        raise ValueError(
            "Currently, only 20-second videos are supported"
        )

    prompt = f"""
Create a short romantic cinematic video plan.

Theme: {theme}

Duration: 20 seconds.
Exactly 4 scenes, each lasting 5 seconds.
Vertical video format: 9:16.

Use the same two Indian adult characters throughout:
- Woman: Young Indian woman, long black hair, pastel blue saree.
- Man: Young Indian man, white shirt, dark trousers.

Maintain identical faces, clothing and appearance.

Create a connected romantic story with a beginning,
emotional development and a memorable ending.

Return ONLY valid JSON using this structure:

{{
    "title": "చెప్పలేని ప్రేమ",
    "style": "Telugu romantic cinema",
    "characters": [
        {{
            "id": "woman",
            "description": "Young Indian woman, long black hair, pastel blue saree"
        }},
        {{
            "id": "man",
            "description": "Young Indian man, white shirt, dark trousers"
        }}
    ],
    "scenes": [
        {{
            "scene_number": 1,
            "start": 0,
            "end": 5,
            "description": "Short scene description",
            "image_prompt": "Short English cinematic image prompt",
            "motion_prompt": "Subtle visual movement",
            "camera": "Slow camera movement",
            "lighting": "Soft cinematic lighting"
        }}
    ]
}}

Rules:
- Exactly four scenes.
- Each scene lasts five seconds.
- Use English for all descriptions.
- Keep every field short.
- Reuse the same characters in every scene.
- No explanations or Markdown.
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1200,
            "num_ctx": 2048
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request, timeout=300
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        generated_text = result.get("response", "")

        print("\n--- OLLAMA DEBUG ---")
        print("Done reason:", result.get("done_reason"))
        print("Generated characters:", len(generated_text))
        print("--- END DEBUG ---\n")

        if not generated_text:
            raise ValueError("Ollama returned an empty response")

        if result.get("done_reason") == "length":
            raise ValueError(
                "Model reached its token limit. "
                "Try a shorter prompt or higher token limit."
            )

        story = json.loads(generated_text)

        return validate_story(story)

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama: {error}"
        ) from error

    except json.JSONDecodeError as error:
        raise ValueError(
            f"The model returned invalid JSON: {error}"
        ) from error



if __name__ == "__main__":

    try:
        story = generate_story(
            theme="A shy couple meets during a rainy evening in Hyderabad.",
            duration=20
        )

        STORY_OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            STORY_OUTPUT,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                story,
                file,
                indent=2,
                ensure_ascii=False
            )

        print(f"\nStory saved successfully: {STORY_OUTPUT}")

    except Exception as error:
        print(f"\nStory generation failed: {error}")

