import torch

from neuralsignal.inference.masking import masked_mean, masked_std, masked_var


def test_masked_mean_ignores_padding_tokens():
    values = torch.tensor([
        [[1.0], [3.0], [999.0]],
        [[2.0], [4.0], [6.0]],
    ])
    mask = torch.tensor([
        [1, 1, 0],
        [1, 1, 1],
    ])

    result = masked_mean(values, mask)

    assert torch.allclose(result, torch.tensor([[2.0], [4.0]]))


def test_masked_var_ignores_padding_tokens():
    values = torch.tensor([
        [[1.0], [3.0], [999.0]],
    ])
    mask = torch.tensor([[1, 1, 0]])

    result = masked_var(values, mask)

    assert torch.allclose(result, torch.tensor([[1.0]]))


def test_masked_std_matches_unpadded_prompt_when_batched_with_longer_prompt():
    short_alone = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    short_batched = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [999.0, 999.0], [999.0, 999.0]],
        [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]],
    ])

    alone_mask = torch.tensor([[1, 1]])
    batch_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    assert torch.allclose(masked_mean(short_alone, alone_mask)[0], masked_mean(short_batched, batch_mask)[0])
    assert torch.allclose(masked_std(short_alone, alone_mask)[0], masked_std(short_batched, batch_mask)[0])

