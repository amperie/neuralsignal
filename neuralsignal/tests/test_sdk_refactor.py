import yaml
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from neuralsignal.core.modules.detector import DetectionResults as DetectorResult
from neuralsignal.core.modules.detector import Detector
from neuralsignal.core.modules.generation_instance import GenerationInstance
from neuralsignal.config.loader import load_sdk_config
from neuralsignal.sdk.neuralsignal import EvaluationError, SDK


BASE_CONFIG = {
    "home": "./sdk_home_test",
    "logging_level": "INFO",
    "evaluation_mode": "indirect",
    "application_name": "default",
    "sub_application_name": "default",
    "max_new_tokens": 1,
    "truncation_length": 0,
    "use_dynamic_batch_size": False,
    "max_oom_count": 2,
    "zone_size": 64,
    "indirect_config": {
        "indirect_model": "stub-model",
        "zone_size": 64,
        "indirect_batch_size": 1,
        "quantization": "no_quantization",
        "device": "cpu",
        "use_dynamic_batch_size": False,
        "max_oom_count": 2,
    },
    "indirect_instrumentation_config": {
        "instrument_encoder": True,
        "instrument_decoder": True,
        "instrument_attention": True,
        "instrument_FF": True,
        "instrument_embedding": True,
        "collector_config": {
            "mode": "additive",
            "data_to_save": ["outputs"],
            "zone_size": 64,
            "zone_size_by_layer": {"default": 64},
            "layer_names_to_include": ["all"],
            "layer_indexes_to_include": [],
            "abort_on_layer_index": 0,
        },
    },
    "save_scans": False,
    "save_vectors": False,
    "backend_config": {
        "backend_type": "file_backend",
    },
    "detectors": [
        {
            "behavior_name": "custom-detector",
            "type": "normal",
            "S1_model": None,
            "S1_model_path": None,
            "prompt": "Prompt {input}",
            "enabled": True,
            "enable_prediction": False,
        }
    ],
}


@pytest.fixture
def config_path() -> Path:
    base_dir = Path("neuralsignal/tests/_tmp")
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"sdk_config_{uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(BASE_CONFIG), encoding="utf-8")
    try:
        yield path
    finally:
        if path.exists():
            path.unlink()


def test_detector_uses_direct_model_instance(monkeypatch):
    class FakeBackend:
        def __init__(self, cfg):
            self.cfg = cfg

        def load_s1_model(self, path):
            raise AssertionError("backend should not load a model")

    monkeypatch.setattr("neuralsignal.core.modules.detector.NSBackend", FakeBackend)

    direct_model = object()
    detector = Detector({
        "application_name": "app",
        "sub_application_name": "sub",
        "S1_model_instance": direct_model,
        "enabled": True,
        "enable_prediction": True,
    })

    assert detector.model is direct_model


def test_sdk_deep_merges_nested_config_and_uses_instance_detectors(monkeypatch, config_path):
    created = []

    class FakeDetector:
        def __init__(self, cfg):
            created.append(cfg)
            self.cfg = cfg

    monkeypatch.setattr("neuralsignal.sdk.neuralsignal.load_model", lambda cfg: ("tokenizer", SimpleNamespace(device="cpu")))
    monkeypatch.setattr("neuralsignal.sdk.neuralsignal.Detector", FakeDetector)

    sdk = SDK(
        application_name="app",
        sub_application_name="sub",
        config={
            "save_scans": False,
            "indirect_config": {"device": "meta"},
            "detectors": [{
                "behavior_name": "custom-detector",
                "type": "normal",
                "S1_model": None,
                "S1_model_path": None,
                "prompt": "Overridden {input}",
                "enabled": True,
                "enable_prediction": False,
            }],
        },
        default_config_path=str(config_path),
    )
    sdk.evaluate_indirect_output = lambda outputs, detectors: detectors

    detectors = sdk.evaluate_indirect([
        {"input": "hello", "output": "world"}
    ], ["custom-detector"])

    assert sdk.cfg["indirect_config"]["device"] == "meta"
    assert sdk.cfg["indirect_config"]["quantization"] == "no_quantization"
    assert detectors[0].cfg["prompt"] == "Overridden {input}"
    assert created[0]["application_name"] == "app"
    assert created[0]["sub_application_name"] == "sub"


def test_evaluate_batch_output_raises_evaluation_error(monkeypatch, config_path):
    monkeypatch.setattr("neuralsignal.sdk.neuralsignal.load_model", lambda cfg: ("tokenizer", SimpleNamespace(device="cpu")))
    monkeypatch.setattr("neuralsignal.sdk.neuralsignal.wrap_with_prompt", lambda prompt, output: "prompt")
    monkeypatch.setattr(
        "neuralsignal.sdk.neuralsignal.generate_from_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    sdk = SDK(
        application_name="app",
        sub_application_name="sub",
        config={"save_scans": False},
        default_config_path=str(config_path),
    )
    detector = SimpleNamespace(prompt="Prompt {input}", type="normal", enabled=True)

    with pytest.raises(EvaluationError):
        sdk._evaluate_batch_output([
            {"input": "hello", "output": "world"}
        ], [detector])


def test_evaluate_indirect_output_returns_public_detection_mapping(monkeypatch, config_path):
    monkeypatch.setattr("neuralsignal.sdk.neuralsignal.load_model", lambda cfg: ("tokenizer", SimpleNamespace(device="cpu")))

    sdk = SDK(
        application_name="app",
        sub_application_name="sub",
        config={"save_scans": False},
        default_config_path=str(config_path),
    )

    gi = GenerationInstance({
        "input": "hello",
        "output": "world",
        "ground_truth": "world",
    })
    gi.data["metadata"] = {"row": 1}
    gi.data["generation_correlation_id"] = "corr-1"
    gi.add_detection(DetectorResult("custom-detector", [0.1, 0.9]))

    monkeypatch.setattr(sdk, "_evaluate_batch_output", lambda outputs, detectors: [gi])

    results = sdk.evaluate_indirect_output([
        {"input": "hello", "output": "world"}
    ], [SimpleNamespace()])

    assert isinstance(results[0].detections, dict)
    assert "custom-detector" in results[0].detections
    assert results[0].correlation_id == "corr-1"




def test_load_sdk_config_normalizes_detector_registry_and_overrides():
    resolved = load_sdk_config(overrides={
        "home": "./sdk_home_test_overrides",
        "detectors": {
            "custom-detector": {
                "prompt": "Mapped {input}",
                "enabled": True,
                "enable_prediction": False,
            },
            "second": {
                "behavior_name": "second",
                "prompt": "Two",
                "enabled": False,
                "enable_prediction": False,
            },
        },
        "backend_config": {"backend_type": "noop"},
    })

    detector_cfg = resolved.get_detector_config(
        "custom-detector",
        overrides={"prompt": "Per call {input}"},
    )

    assert set(resolved.detector_registry.keys()) == {"custom-detector", "second"}
    assert detector_cfg["prompt"] == "Per call {input}"
    assert resolved.get_backend_config()["backend_type"] == "noop"


def test_detector_uses_explicit_backend_override(monkeypatch):
    captured = {}

    class FakeBackend:
        def __init__(self, cfg):
            captured.update(cfg)

        def load_s1_model(self, path):
            raise AssertionError("backend should not load a model")

    monkeypatch.setattr("neuralsignal.core.modules.detector.NSBackend", FakeBackend)

    Detector({
        "application_name": "app",
        "sub_application_name": "sub",
        "backend_config": {"backend_type": "noop"},
        "enabled": False,
        "enable_prediction": False,
    })

    assert captured["backend_config"]["backend_type"] == "noop"
    assert captured["application_name"] == "app"
