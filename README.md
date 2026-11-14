# HeyFunverse AI

Local AI pipeline that turns a Telugu or English story idea into a vertical (9:16) Instagram Reel:
story plan, scene images, motion, captions, music and final video.
Everything runs on your own machine (Windows laptop, no NVIDIA GPU) with
Ollama (Qwen 2.5 3B), OpenVINO Stable Diffusion 1.5 and FFmpeg.

## Status

| Stage | Module | State |
| --- | --- | --- |
| Story plan | `ai-engine/story_planner.py` | works (Qwen via Ollama) |
| Scene jobs | `ai-engine/scene_processor.py` | works |
| Scene images | `ai-engine/image_generator.py` | works; prompts come from the story |
| Motion | `ai-engine/video_generator.py` | not built yet |
| Render (captions, music, transitions) | `ai-engine/video_renderer.py` | not built yet |
| Orchestration | `ai-engine/workflow_manager.py` | story to scene jobs only |

Support tools: `config.py` (all settings), `safe_io.py` (backups before any overwrite),
`check_env.py` (environment doctor), `benchmark.py` (CPU/GPU/AUTO timing),
`face_refine.py` and `fetch_face_models.py` (optional face tools).

## Windows setup

Requirements: Windows 10/11, Python 3.13 (python.org), [Ollama](https://ollama.com) with
`ollama pull qwen2.5:3b`, and FFmpeg on PATH (an "essentials" or "full" build from gyan.dev
includes libass and HarfBuzz, which Telugu captions need later).

```powershell
git clone https://github.com/BJC93/heyfunverse-ai
cd heyfunverse-ai
.\scripts\setup_windows.ps1 -Dev     # creates .venv, installs packages, runs the environment check
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script: `Set-ExecutionPolicy -Scope Process Bypass`.
The setup downloads Python packages only. The Stable Diffusion model (several GB) is downloaded
and converted the first time you generate an image, and only then.

Reproducible installs: once everything works, run `.\scripts\freeze_env.ps1` and keep
`requirements.lock.txt`; on another machine use `.\scripts\setup_windows.ps1 -Lock`.

Check the machine at any time:

```powershell
python ai-engine\check_env.py          # add --json for machine-readable output
```

## Usage

```powershell
python ai-engine\workflow_manager.py --generate --theme "A shy couple meets in the rain."   # Qwen writes storage\story.json
python ai-engine\workflow_manager.py                                                       # story -> ai-engine\scene_jobs.json
python ai-engine\image_generator.py --dry-run                                              # preview prompts, loads nothing
python ai-engine\image_generator.py --scene 1                                              # one image
python ai-engine\image_generator.py                                                        # all missing images
```

Images are written to `storage\images\scene_01.png` (432x768) with a `scene_01.json` next to each
(prompt, seed, steps, timing). Existing images are skipped; `--force` regenerates them.

## Safety rules the tools follow

- Nothing replaces an existing story, scene plan or image without first copying the old file to
  `storage\backups\` (see `safe_io.py`). Identical content is not rewritten.
- No model is downloaded except the image model on first use and the optional small face models,
  which need an explicit `--yes`.
- Nothing is committed, deleted or pushed automatically.

## Settings

All settings live in `ai-engine/config.py` and can be overridden with environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HEYFUN_DEVICE` | `CPU` | `CPU`, `GPU` or `AUTO` (GPU/AUTO unproven until you run the benchmark) |
| `HEYFUN_STEPS` | `30` | inference steps per image |
| `HEYFUN_SEED` | `1234` | base seed (scene number is added) |
| `HEYFUN_IMAGE_MODEL` | `SG161222/Realistic_Vision_V5.1_noVAE` | Hugging Face model id (`IMAGE_MODEL_ID` also works) |
| `HEYFUN_OLLAMA_MODEL` | `qwen2.5:3b` | local planning model |
| `HEYFUN_OLLAMA_URL` | `http://localhost:11434` | Ollama address |
| `HEYFUN_STORAGE_DIR` | `storage` | where stories, images, backups live |

## Benchmark

Run after at least one image has been generated (the benchmark never downloads models).
Plug in the charger and close other heavy apps first.

```powershell
python ai-engine\benchmark.py --devices CPU GPU AUTO --images 2 --steps 20
python ai-engine\benchmark.py --ollama --skip-image                  # Qwen tokens per second
python ai-engine\benchmark.py --check-refiner-memory                 # face refiner must not add a model copy
```

Each device runs in its own process, so a GPU failure only fails that row. Results are saved to
`storage\benchmarks\` as Markdown and JSON. A device that OpenVINO lists is not assumed to work.

## Tests

```powershell
python -m pytest
```

The tests use fake pipelines, so they run in seconds without Ollama, OpenVINO or any model.

## Optional face tools

See `docs/FACE_MODELS.md`. Face refinement is off by default (`image_generator.py --refine-faces`)
and reuses the already loaded model; it never loads a second copy.

## Roadmap

Phase 0 (this baseline) then Phase 1 (project.json, flexible scene counts, Telugu/English input),
Phase 2 (consistent characters and image review), Phase 3 (motion and render), Phase 4 (Claude Code
agent layer), Phase 5 (backend and UI).
