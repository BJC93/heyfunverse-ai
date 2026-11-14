import json
import subprocess
import sys
from pathlib import Path

import check_env as ce

ROOT = Path(__file__).resolve().parents[1]


def test_buildconf_parser():
    f = ce.parse_ffmpeg_buildconf("--enable-libass --enable-libharfbuzz --enable-libx264")
    assert f == {"libass": True, "harfbuzz": True, "fribidi": False, "x264": True}


def test_ffmpeg_missing():
    assert ce.check_ffmpeg(which=lambda n: None)[0].status == ce.FAIL


def test_ffmpeg_features_reported():
    class Done:
        def __init__(self, out): self.stdout, self.stderr = out, ""
    def run(cmd, **kw):
        return Done("ffmpeg version 7.1\n" if "-version" in cmd else "--enable-libass --enable-libx264")
    res = {c.name: c.status for c in ce.check_ffmpeg(run=run, which=lambda n: "ffmpeg")}
    assert res["FFmpeg libass (subtitles)"] == ce.OK and res["FFmpeg HarfBuzz (Telugu shaping)"] == ce.WARN


def test_ollama_unreachable_is_a_warning_not_failure():
    import urllib.error
    def boom(*a, **k): raise urllib.error.URLError("refused")
    assert ce.check_ollama(opener=boom)[0].status == ce.WARN


def test_ollama_model_found():
    import io
    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass
    body = json.dumps({"models": [{"name": "qwen2.5:3b"}]}).encode()
    assert ce.check_ollama(opener=lambda *a, **k: R(body))[0].status == ce.OK


def test_exit_code_and_report_format():
    res = [ce.Check("a", ce.OK, "x"), ce.Check("b", ce.FAIL, "y")]
    assert ce.exit_code(res) == 1 and ce.exit_code(res[:1]) == 0
    assert "[FAIL] b" in ce.format_report(res) and "1 failed" in ce.format_report(res)


def test_cli_json_output_is_valid():
    out = subprocess.run([sys.executable, str(ROOT / "ai-engine" / "check_env.py"), "--json"],
                         capture_output=True, text=True)
    data = json.loads(out.stdout)
    assert any(c["name"] == "Python" for c in data)
