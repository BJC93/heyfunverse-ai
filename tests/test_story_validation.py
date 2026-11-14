import pytest

import scene_processor
import story_planner
from conftest import make_story


def test_valid_story_passes(story):
    assert story_planner.validate_story(story)["scenes"][0]["scene_number"] == 1


def test_story_must_be_object():
    with pytest.raises(ValueError):
        story_planner.validate_story([])


def test_missing_scene_field_is_reported(story):
    del story["scenes"][2]["camera"]
    with pytest.raises(ValueError, match="camera"):
        story_planner.validate_story(story)


def test_current_default_is_exactly_four_scenes():
    # Phase 1 generalises this; the test pins today's behaviour.
    with pytest.raises(ValueError):
        story_planner.validate_story(make_story(3))


def test_scene_plan_validates(story):
    scene_processor.validate_scene_plan(story)


def test_jobs_follow_the_story(story):
    jobs = scene_processor.prepare_scene_jobs(story)
    assert [j["scene_number"] for j in jobs] == [1, 2, 3, 4]
    assert all(j["duration"] == 5 for j in jobs)
    assert jobs[1]["image_prompt"].startswith("Scene 2")


def test_planner_uses_config_values():
    import config
    assert story_planner.STORY_OUTPUT == config.STORY_FILE
    assert story_planner.MODEL == config.OLLAMA_MODEL
