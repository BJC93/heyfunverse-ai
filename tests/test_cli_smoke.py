import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_prints_a_prompt_per_scene_without_loading_a_model():
    out = subprocess.run([sys.executable, str(ROOT / "ai-engine" / "image_generator.py"), "--dry-run"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("Scene ") >= 4
    assert "DRY RUN" in out.stdout


def test_dry_run_single_scene():
    out = subprocess.run([sys.executable, str(ROOT / "ai-engine" / "image_generator.py"), "--dry-run", "--scene", "2"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and "Scene 2" in out.stdout and "Scene 1" not in out.stdout
