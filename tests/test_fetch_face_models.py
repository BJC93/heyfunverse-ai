import io

import pytest

import fetch_face_models as ff


def test_size_bounds():
    assert ff.size_ok(233_000, (150_000, 400_000))
    assert not ff.size_ok(5_000, (150_000, 400_000))


def test_dry_run_downloads_nothing(capsys):
    assert ff.main([]) == 0
    assert "Dry run" in capsys.readouterr().out


class Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): pass


def test_download_accepts_expected_size(tmp_path):
    dest = tmp_path / "m.onnx"
    size, digest = ff.download("u", dest, (10, 100), opener=lambda *a, **k: Resp(b"x" * 50))
    assert size == 50 and len(digest) == 64 and dest.exists()


def test_download_rejects_wrong_size_and_leaves_nothing(tmp_path):
    dest = tmp_path / "m.onnx"
    with pytest.raises(ValueError):
        ff.download("u", dest, (10, 100), opener=lambda *a, **k: Resp(b"x" * 5000))
    assert not dest.exists() and not list(tmp_path.glob("*.part"))
