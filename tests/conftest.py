import copy

import pytest

SCENE_FIELDS = {
    "description": "Woman walks in the rain, man watches from behind her.",
    "image_prompt": "Woman in pastel blue saree walking on a rainy street, man in white shirt far behind her",
    "motion_prompt": "Gentle rain, woman walks slowly.",
    "camera": "Medium shot of the woman, then close-up of the man.",
    "lighting": "Soft street lights, wet road reflections.",
}


def make_story(scene_count=4):
    scenes = []
    for number in range(1, scene_count + 1):
        scene = dict(SCENE_FIELDS)
        scene.update(
            scene_number=number,
            start=(number - 1) * 5,
            end=number * 5,
            image_prompt=f"Scene {number}: " + SCENE_FIELDS["image_prompt"],
        )
        scenes.append(scene)

    return {
        "title": "చెప్పలేని ప్రేమ",
        "style": "Telugu romantic cinema",
        "characters": [
            {"id": "woman", "description": "Young Indian woman, long black hair, pastel blue saree"},
            {"id": "man", "description": "Young Indian man, white shirt, dark trousers"},
        ],
        "scenes": scenes,
    }


@pytest.fixture
def story():
    return copy.deepcopy(make_story())
