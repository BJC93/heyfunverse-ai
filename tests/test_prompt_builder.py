import config
import image_generator as ig
from conftest import make_story


def jobs_for(story):
    import scene_processor
    return scene_processor.prepare_scene_jobs(story)


def test_prompt_comes_from_the_scene_not_a_table(story):
    jobs = jobs_for(story)
    prompts = [ig.build_prompt(j) for j in jobs]
    assert len(set(prompts)) == 4
    assert prompts[2].startswith("Scene 3")


def test_new_story_text_reaches_the_prompt(story):
    story["scenes"][0]["image_prompt"] = "A tea stall in Hyderabad at dawn"
    prompt = ig.build_prompt(jobs_for(story)[0])
    assert "tea stall in Hyderabad" in prompt


def test_character_match_uses_whole_words(story):
    job = jobs_for(story)[0]
    job["image_prompt"] = "Woman in a saree on a street"
    job["description"] = "Woman walks"
    ids = [c["id"] for c in ig.characters_in_scene(job)]
    assert ids == ["woman"]          # "man" inside "woman" must not match


def test_visual_identity_wins_over_legacy_and_description():
    char = {"id": "woman", "description": "plain", "visual_identity": "tall woman, red saree"}
    assert ig.character_look(char) == "tall woman, red saree"


def test_unknown_character_falls_back_to_description():
    assert ig.character_look({"id": "kid", "description": "small boy, yellow cap"}) == "small boy, yellow cap"


def test_legacy_anchor_used_when_no_visual_identity():
    assert "bindi" in ig.character_look({"id": "woman", "description": "x"})


def test_prompt_fits_token_budget(story):
    for job in jobs_for(story):
        prompt, _ = ig.build_prompt_with_report(job)
        assert ig.approx_tokens(prompt) <= config.MAX_PROMPT_TOKENS


def test_least_important_parts_are_dropped_first():
    parts = [("scene", "a b c"), ("character:x", "d e f"), ("quality", "g h i"), ("lighting", "j k l")]
    count = lambda t: len(t.split())
    prompt, dropped = ig.fit_prompt(parts, count, max_tokens=8)
    assert prompt.startswith("a b c") and "d e f" in prompt
    assert dropped == ["quality", "lighting"] or dropped == ["lighting"]
    assert "j k l" not in prompt


def test_scene_part_is_always_kept_even_if_too_long():
    prompt, dropped = ig.fit_prompt([("scene", "one two three four five")], lambda t: len(t.split()), 2)
    assert prompt == "one two three four five" and dropped == []


def test_first_clause():
    assert ig.first_clause("Wide shot of rain, then close-up of hands.") == "Wide shot of rain"


def test_select_jobs():
    jobs = [{"scene_number": 1}, {"scene_number": 2}]
    assert ig.select_jobs(jobs, 2) == [{"scene_number": 2}]
    assert ig.select_jobs(jobs) == jobs
    import pytest
    with pytest.raises(ValueError):
        ig.select_jobs(jobs, 9)


def test_center_crop_gives_9_16():
    from PIL import Image
    image = ig.center_crop(Image.new("RGB", (512, 768)), config.CROP_SIZE)
    assert image.size == (432, 768)


def test_quality_words_survive_a_tight_budget_before_character_looks(story):
    job = jobs_for(story)[0]
    names = [n for n, _ in ig.build_prompt_parts(job)]
    assert names.index("quality") < names.index("character:woman")
    parts = dict(ig.build_prompt_parts(job))
    budget = ig.approx_tokens(parts["scene"]) + ig.approx_tokens(config.QUALITY_SUFFIX) + 2
    prompt, dropped = ig.build_prompt_with_report(job, max_tokens=budget)
    assert config.QUALITY_SUFFIX in prompt
    assert "character:woman" in dropped
