import torch

from neuralsignal.core.modules.feature_sets.feature_set_layer_distribution import FeatureSetLayerDistribution
from neuralsignal.core.modules.feature_sets.feature_set_zones import FeatureSetZones
from neuralsignal.core.modules.feature_sets.feature_utils import apply_attention_mask


def test_apply_attention_mask_filters_token_rows():
    tensor = torch.tensor([[1.0], [2.0], [999.0]])
    mask = torch.tensor([1, 1, 0])

    result = apply_attention_mask(tensor, mask)

    assert result.tolist() == [[1.0], [2.0]]


def test_layer_distribution_ignores_padding_outlier():
    scan = {
        "layer_order": ["l0"],
        "layer_id_to_name": {"l0": "Dense"},
        "outputs": {"l0": torch.tensor([[1.0], [1.0], [999.0]])},
        "inputs": {"l0": torch.tensor([[0.0], [0.0], [0.0]])},
        "attention_mask": torch.tensor([1, 1, 0]),
    }
    fs = FeatureSetLayerDistribution({
        "layers_to_process": ["Dense"],
        "field_to_process": "outputs",
        "bin_count": 2,
        "output_format": "name_and_value_columns",
    })

    _, values = fs.process_feature_set(scan)

    assert sum(values) == 2.0


def test_zones_mean_ignores_padding_outlier():
    scan = {
        "layer_order": ["l0"],
        "layer_id_to_name": {"l0": "Dense"},
        "zone_size": 1,
        "outputs": {"l0": torch.tensor([[1.0], [3.0], [999.0]])},
        "attention_mask": torch.tensor([1, 1, 0]),
    }
    fs = FeatureSetZones({
        "target_zone_size": {"default": 1},
        "field_to_process": "outputs",
        "layer_names_to_include": ["Dense"],
        "layer_indexes_to_include": None,
        "output_format": "name_and_value_columns",
    })

    _, values = fs.process_feature_set(scan)

    assert values == [2.0]



def test_true_false_diff_uses_last_unpadded_token():
    from neuralsignal.core.modules.feature_sets.feature_set_t_f_diff import FeatureSetTrueFalseDiff

    class FakeUnembed:
        def forward(self, t):
            logits = torch.zeros((t.shape[0], 11000))
            logits[:, 10998] = t[:, 0]
            logits[:, 10747] = t[:, 1]
            return logits

    scan = {
        "layer_id_to_name": {"l0": "Dense"},
        "outputs": {"l0": torch.tensor([[1.0, 0.0], [4.0, 1.0], [999.0, 999.0]])},
        "attention_mask": torch.tensor([1, 1, 0]),
    }
    fs = FeatureSetTrueFalseDiff({
        "layers_to_process": ["Dense"],
        "unembed_layer": FakeUnembed(),
        "dev_map": "cpu",
        "output_format": "name_and_value_columns",
    })

    _, values = fs.process_feature_set(scan)

    assert values[0] == 3.0
