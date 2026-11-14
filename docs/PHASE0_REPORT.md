# Phase 0 report

Phase 0 goal: a reproducible baseline with real measurements. Everything below was built in a
staging copy of `BJC93/heyfunverse-ai` (commit `a94d74d`). No existing file was modified in your
working tree, nothing was committed, and no model was downloaded.

## Results

- 74 automated tests pass, 2 are skipped because the sandbox's OpenCV 5.0 build has no Haar cascade
  files (`python -m pytest`, about 1 second, no models needed).
- `image_generator.py --dry-run` prints the prompt for every scene from the story data and loads no model.
- Not measured: the CPU/GPU/AUTO benchmark and Ollama speed. They need your laptop; the tools are
  ready (`benchmark.py`). The Windows PowerShell scripts were written but could not be run here.
- Not verified: pip installs (the sandbox has no package network), real OpenVINO behaviour, the
  actual face-model downloads.

## Proposed changes to EXISTING files (review `phase0-existing-files.patch`)

| File | Change | Risk |
| --- | --- | --- |
| `ai-engine/image_generator.py` | Rewritten: prompts built from scene data with a 77-token budget; settings from `config.py`; `--dry-run`; `--refine-faces` (off by default, reuses loaded model); old image backed up before `--force` overwrite; metadata file per image | Medium: biggest diff. Same engine, sizes and file names. |
| `ai-engine/story_planner.py` | Defines the missing `STORY_OUTPUT`; settings from `config.py`; story saved through backup-first writer | Low |
| `ai-engine/workflow_manager.py` | `--generate` backs up `storage/story.json` before replacing it; new `--theme` option | Low |
| `.gitignore` | Adds `/storage/backups/` and `.pytest_cache/` | None |
| `README.md` (empty today) | Setup, usage, safety rules, settings, benchmark, tests | None |

`scene_processor.py`, `story.json`, `scene_jobs.json` and the empty video modules are untouched.

## New files (`phase0-new-files.zip`)

`ai-engine/`: `config.py`, `safe_io.py`, `check_env.py`, `benchmark.py`, `face_refine.py`,
`fetch_face_models.py`. Also `requirements.txt`, `requirements-dev.txt`, `pytest.ini`,
`scripts/setup_windows.ps1`, `scripts/freeze_env.ps1`, `docs/FACE_MODELS.md`, `docs/ASSETS.md`,
`tests/` (10 files).

## Findings that affect Phase 1 and 2

1. Prompt budget is tight. With the revised story, scene text (about 35 tokens) plus two character
   descriptions leaves little room: the man's description, camera and lighting are dropped in most
   scenes. Phase 1 should shorten scene prompts and character identities at planning time.
2. The face refiner assembles an img2img pipeline from the loaded components. I could not verify
   that optimum-intel allows this, so it is off by default and fails safe (it disables itself and
   never loads a second copy). Run `check_env.py --probe-refiner` and
   `benchmark.py --check-refiner-memory` on your laptop to find out.
3. OpenCV 5 builds may not ship Haar cascade files. The detector falls back cleanly, and YuNet is
   the more reliable choice once its small model is installed.
4. `requirements.txt` is unpinned on purpose; `freeze_env.ps1` records the versions that work for you.
5. The legacy look anchors for "woman" and "man" in `config.py` remain only as a fallback for
   stories without `visual_identity`; Phase 1's character bible replaces them.

## Needs your action before Phase 1

1. Review and apply the patch, unzip the new files, run `python -m pytest`.
2. Run `python ai-engine\check_env.py` and send me the output.
3. Run `python ai-engine\benchmark.py --devices CPU GPU AUTO --images 2 --steps 20` and, separately,
   `python ai-engine\benchmark.py --ollama --skip-image`; send the Markdown results from `storage\benchmarks\`.
4. Optionally run `check_env.py --probe-refiner` and `fetch_face_models.py --yes` then `--probe-face`.
