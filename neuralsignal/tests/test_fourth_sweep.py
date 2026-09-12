from types import SimpleNamespace

import pytest
import torch

from neuralsignal.config import load_config
from neuralsignal.core.modules.collector import Collector
from neuralsignal.core.modules.tensors import process_tensor_dict_into_zones_by_layer


@pytest.mark.parametrize('zone_map', [{}, {'Dense': 2}])
def test_collector_uses_global_zone_default(zone_map):
    collector = Collector({'zone_size': 4, 'zone_size_by_layer': zone_map})
    assert collector.zone_size == 4


def test_collector_filters_layers_without_zone_overrides():
    collector = Collector({'zone_size_by_layer': {}, 'zone_size': 1,
                           'layer_names_to_include': ['keep'], 'data_to_save': ['outputs', 'layer_info']})
    for name in ['keep', 'drop']:
        module = SimpleNamespace(ns_name=name)
        collector(module, (torch.ones(1, 2, 4),), torch.ones(1, 2, 4))
    collector.finish_and_get_data()
    scan = collector.get_data_by_batch_index(0)
    assert [scan['layer_id_to_name'][key] for key in scan['outputs']] == ['keep']


def test_collector_supports_input_only_topology():
    collector = Collector({'zone_size_by_layer': {'default': 1},
                           'data_to_save': ['inputs', 'layer_info', 'topology']})
    collector(SimpleNamespace(ns_name='Dense'), (torch.ones(1, 2, 4),), torch.ones(1, 2, 4))
    collector.finish_and_get_data()
    assert collector.get_data_by_batch_index(0)['topology'] == [[2, 4]]


def test_per_layer_reduction_rejects_fractional_ratio():
    with pytest.raises(ValueError, match='multiple'):
        process_tensor_dict_into_zones_by_layer(
            {'l0': torch.ones(2, 4)}, {'default': 3}, {'l0': 'Dense'}, {'default': 2}, 3,
            layer_indexes_to_include=[], layer_names_to_include=[])


def test_smoke_settings_through_real_t5_collector_and_features():
    from transformers import T5Config, T5ForConditionalGeneration
    from neuralsignal.core.modules.model_instrumentation import generate_from_batch
    from neuralsignal.features.model_extractor import ModelFeatureExtractor
    from neuralsignal.features.selection import materialized_feature_sets
    from neuralsignal.datasets.v2 import DatasetExample

    torch.manual_seed(7)
    config = load_config('configs/feature_collection/smoke_runpod_malt.yaml')
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=32, d_model=16, d_kv=4, d_ff=32, num_layers=1,
        num_decoder_layers=1, num_heads=2, dropout_rate=0,
        decoder_start_token_id=0, eos_token_id=1, pad_token_id=0,
    )).eval()
    class Tokenizer:
        eos_token_id = 1
        def __call__(self, texts, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[2, 3, 0], [2, 3, 4]]),
                                   attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]))
        def decode(self, *args, **kwargs):
            return 'response'
    scans = generate_from_batch(['short', 'long'], model, Tokenizer(),
                                config['instrumentation'], truncation_length=512, max_new_tokens=2)
    extractor = ModelFeatureExtractor(config)
    rows = [extractor._featurize(scan.data, DatasetExample(str(i), 'i', 'o'), materialized_feature_sets(config))
            for i, scan in enumerate(scans)]
    assert len(rows) == 2
    for row in rows:
        assert any(key.startswith('zones__') for key in row)
        assert any(key.startswith('layer_distribution__') for key in row)
        assert all(torch.isfinite(torch.tensor(value)) for value in row.values())
    assert rows[0].keys() == rows[1].keys()
    assert all(not module._forward_hooks for module in model.modules())
