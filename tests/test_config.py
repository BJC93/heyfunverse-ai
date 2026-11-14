import importlib

import pytest

import config


def reload_config():
    return importlib.reload(config)


def test_defaults(monkeypatch):
    for name in ("HEYFUN_DEVICE", "HEYFUN_STEPS", "HEYFUN_OLLAMA_MODEL", "HEYFUN_IMAGE_MODEL", "IMAGE_MODEL_ID"):
        monkeypatch.delenv(name, raising=False)
    cfg = reload_config()
    assert cfg.DEFAULT_DEVICE == "CPU"
    assert cfg.INFERENCE_STEPS == 30
    assert cfg.OLLAMA_MODEL == "qwen2.5:3b"
    assert (cfg.WIDTH, cfg.HEIGHT) == (512, 768)
    assert cfg.CROP_SIZE == (432, 768)


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("HEYFUN_DEVICE", "auto")
    monkeypatch.setenv("HEYFUN_STEPS", "12")
    cfg = reload_config()
    assert cfg.DEFAULT_DEVICE == "AUTO"
    assert cfg.INFERENCE_STEPS == 12


def test_bad_device_is_rejected(monkeypatch):
    monkeypatch.setenv("HEYFUN_DEVICE", "TPU")
    with pytest.raises(ValueError):
        reload_config()


def test_bad_integer_is_rejected(monkeypatch):
    monkeypatch.setenv("HEYFUN_STEPS", "many")
    with pytest.raises(ValueError):
        reload_config()


def test_legacy_model_variable_still_works(monkeypatch):
    monkeypatch.delenv("HEYFUN_IMAGE_MODEL", raising=False)
    monkeypatch.setenv("IMAGE_MODEL_ID", "someone/Some_Model")
    cfg = reload_config()
    assert cfg.IMAGE_MODEL_ID == "someone/Some_Model"
    assert cfg.MODEL_DIR.name == "openvino-some_model"


def teardown_module(module):
    importlib.reload(config)
