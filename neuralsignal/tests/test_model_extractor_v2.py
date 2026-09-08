from types import SimpleNamespace

import pandas as pd
import torch

from neuralsignal.core.modules.generation_instance import GenerationInstance
from neuralsignal.core.modules.model_instrumentation import generate_from_batch
from neuralsignal.datasets.v2 import DatasetExample
from neuralsignal.features.model_extractor import ModelFeatureExtractor
from neuralsignal.features.runner import collect_features_batched
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest


def test_collect_features_batched_writes_one_row_per_example(tmp_path):
    config = {"features": {"materialize": [{"name": "zones"}]}, "generation": {"batch_size": 2}}
    manifest = RunManifest("run-1", dataset={"name": "fixture"}, features={"schema_version": "features.v1"})
    writer = LocalFeatureShardWriter(tmp_path, manifest)
    examples = [DatasetExample(id=str(i), input="i", output="o") for i in range(3)]
    seen_batches = []

    def extractor(batch, specs):
        seen_batches.append([example.id for example in batch])
        return [{"zones__batch_size": len(batch), "zones__specs": len(specs)} for _ in batch]

    collect_features_batched(examples, config, writer, extractor)

    rows = pd.read_parquet(tmp_path / "features" / "part-00000.parquet")
    assert rows["example_id"].tolist() == ["0", "1", "2"]
    assert rows["zones__batch_size"].tolist() == [2.0, 2.0, 1.0]
    assert seen_batches == [["0", "1"], ["2"]]


def test_model_feature_extractor_batches_generation_and_featurizes(monkeypatch):
    calls = {}

    def fake_load_model(model_config):
        calls["model_config"] = model_config
        return object(), SimpleNamespace(name_or_path=model_config["model_name"])

    def fake_generate(prompts, model, tokenizer, instrumentation_cfg, truncation_length=0, max_new_tokens=128):
        calls["prompts"] = prompts
        calls["instrumentation"] = instrumentation_cfg
        calls["truncation_length"] = truncation_length
        calls["max_new_tokens"] = max_new_tokens
        scans = []
        for index, prompt in enumerate(prompts):
            gi = GenerationInstance({"input": prompt, "output": "decoded", "model_name": model.name_or_path})
            gi.add_data({"feature_seed": index + 1})
            scans.append(gi)
        return scans

    class FakeFeatureProcessor:
        def __init__(self, feature_set_configs):
            self.name = feature_set_configs[0]["name"]

        def featurize_scan(self, scan):
            return ([f"{self.name}__seed"], [scan["feature_seed"]])

    import neuralsignal.features.model_extractor as model_extractor

    monkeypatch.setattr(model_extractor, "load_model", fake_load_model)
    monkeypatch.setattr(model_extractor, "generate_from_batch", fake_generate)
    monkeypatch.setattr(model_extractor, "FeatureProcessor", FakeFeatureProcessor)

    extractor = ModelFeatureExtractor({
        "model": {"model_name": "fixture-model", "device": "cpu", "quantization": "none"},
        "generation": {"truncation_length": 64, "max_new_tokens": 3},
        "instrumentation": {"collector_config": {"zone_size": 1}},
        "prompt": {"template": "Q: {input}\nA: {output}"},
    })
    rows = extractor.extract_batch(
        [DatasetExample(id="a", input="i1", output="o1"), DatasetExample(id="b", input="i2", output="o2")],
        [SimpleNamespace(name="zones", config={})],
    )

    assert calls["model_config"]["model_name"] == "fixture-model"
    assert calls["prompts"] == ["Q: i1\nA: o1", "Q: i2\nA: o2"]
    assert calls["truncation_length"] == 64
    assert calls["max_new_tokens"] == 3
    assert rows == [{"zones__seed": 1.0}, {"zones__seed": 2.0}]


def test_generate_from_batch_attaches_attention_masks(monkeypatch):
    class Encoded(SimpleNamespace):
        pass

    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, values, **kwargs):
            assert kwargs["padding"] is True
            return Encoded(
                input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
                attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
            )

        def decode(self, value, skip_special_tokens=True):
            return "decoded"

    class FakeModel:
        device = torch.device("cpu")
        name_or_path = "fixture-model"

        def generate(self, **kwargs):
            assert torch.equal(kwargs["attention_mask"], torch.tensor([[1, 1, 0], [1, 1, 1]]))
            return torch.tensor([[10, 0], [11, 0]])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    scans = generate_from_batch(["short", "longer"], FakeModel(), FakeTokenizer(), instrumentation_cfg=None)

    assert torch.equal(scans[0].data["attention_mask"], torch.tensor([1, 1, 0]))
    assert torch.equal(scans[1].data["attention_mask"], torch.tensor([1, 1, 1]))


def test_model_feature_extractor_rejects_short_generation(monkeypatch):
    import pytest
    import neuralsignal.features.model_extractor as model_extractor

    monkeypatch.setattr(model_extractor, "load_model", lambda cfg: (object(), SimpleNamespace(name_or_path="fixture")))
    monkeypatch.setattr(model_extractor, "generate_from_batch", lambda *args, **kwargs: [])

    extractor = ModelFeatureExtractor({"model": {"model_name": "fixture", "device": "cpu", "quantization": "none"}})

    with pytest.raises(RuntimeError, match="generation returned 0 scans for 1 examples"):
        extractor.extract_batch([DatasetExample(id="a", input="i", output="o")], [SimpleNamespace(name="zones", config={})])
