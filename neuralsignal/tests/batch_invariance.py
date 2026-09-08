from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neuralsignal.core.modules.feature_sets.feature_set_factory import make_feature_set


@dataclass(frozen=True)
class BatchInvarianceResult:
    feature_set: str
    example_index: int
    max_abs_diff: float


class FakeTrueFalseUnembed:
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((t.shape[0], 11000), dtype=t.dtype, device=t.device)
        logits[:, 10998] = t[:, 0]
        logits[:, 10747] = t[:, 1]
        return logits


def assert_batch_invariant(config: dict[str, Any]) -> list[BatchInvarianceResult]:
    lengths = config["test"]["sequence_lengths"]
    scans = [_single_scan(config, index, length) for index, length in enumerate(lengths)]
    padded = _padded_scan(config, lengths)
    feature_configs = [item for item in config["features"]["materialize"] if item.get("enabled", True)]
    results: list[BatchInvarianceResult] = []

    for feature_entry in feature_configs:
        name = feature_entry["name"]
        for index, scan in enumerate(scans):
            single = _feature_dict(name, feature_entry["config"], scan)
            batched = _feature_dict(name, feature_entry["config"], _slice_scan(padded, index))
            keys = set(single) | set(batched)
            if single.keys() != batched.keys():
                missing = sorted(keys - (set(single) & set(batched)))
                raise AssertionError(f"{name} feature keys differ for row {index}: {missing}")
            max_abs_diff = max((abs(single[key] - batched[key]) for key in keys), default=0.0)
            results.append(BatchInvarianceResult(name, index, max_abs_diff))
            _assert_close(config, name, index, max_abs_diff)
    return results


def _feature_dict(name: str, config: dict[str, Any], scan: dict[str, Any]) -> dict[str, float]:
    cfg = dict(config)
    if name == "T-F-diff":
        cfg["unembed_layer"] = FakeTrueFalseUnembed()
    cols, values = make_feature_set(name, cfg).process_feature_set(scan)
    return dict(zip(cols, [float(value) for value in values]))


def _single_scan(config: dict[str, Any], example_index: int, length: int) -> dict[str, Any]:
    layers = config["scan"]["layer_order"]
    names = config["scan"]["layer_names"]
    hidden = int(config["scan"]["hidden_size"])
    outputs = {layer: _activation(example_index, layer_index, length, hidden) for layer_index, layer in enumerate(layers)}
    inputs = {layer: tensor * 0.5 for layer, tensor in outputs.items()}
    return {
        "layer_order": layers,
        "layer_id_to_name": names,
        "zone_size": 1,
        "outputs": outputs,
        "inputs": inputs,
        "attention_mask": torch.ones(length, dtype=torch.long),
    }


def _padded_scan(config: dict[str, Any], lengths: list[int]) -> dict[str, Any]:
    layers = config["scan"]["layer_order"]
    names = config["scan"]["layer_names"]
    hidden = int(config["scan"]["hidden_size"])
    max_len = max(lengths)
    outputs = {}
    inputs = {}
    for layer_index, layer in enumerate(layers):
        rows = []
        for example_index, length in enumerate(lengths):
            activation = _activation(example_index, layer_index, length, hidden)
            pad = torch.full((max_len - length, hidden), 9999.0 + example_index + layer_index)
            rows.append(torch.cat([activation, pad], dim=0))
        outputs[layer] = torch.stack(rows, dim=0)
        inputs[layer] = outputs[layer] * 0.5
    mask = torch.zeros((len(lengths), max_len), dtype=torch.long)
    for index, length in enumerate(lengths):
        mask[index, :length] = 1
    return {
        "layer_order": layers,
        "layer_id_to_name": names,
        "zone_size": 1,
        "outputs": outputs,
        "inputs": inputs,
        "attention_mask": mask,
    }


def _slice_scan(scan: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        **scan,
        "outputs": {layer: tensor[index] for layer, tensor in scan["outputs"].items()},
        "inputs": {layer: tensor[index] for layer, tensor in scan["inputs"].items()},
        "attention_mask": scan["attention_mask"][index],
    }


def _activation(example_index: int, layer_index: int, length: int, hidden: int) -> torch.Tensor:
    base = example_index * 100 + layer_index * 10
    return torch.arange(length * hidden, dtype=torch.float32).reshape(length, hidden) + base


def _assert_close(config: dict[str, Any], name: str, index: int, diff: float) -> None:
    atol = float(config["test"]["tolerance"]["absolute"])
    if diff > atol:
        raise AssertionError(f"{name} row {index} exceeded batch-invariance tolerance: {diff} > {atol}")



def test_configured_batch_invariance() -> None:
    from pathlib import Path

    from neuralsignal.config import load_config

    config = load_config(Path(__file__).parents[2] / "configs" / "tests" / "batch_invariance.yaml")
    assert_batch_invariant(config)
