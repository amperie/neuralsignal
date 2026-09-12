import pytest
import torch
from transformers import LongT5Config, LongT5ForConditionalGeneration

from neuralsignal.core.modules.collector import Collector
from neuralsignal.core.modules.model_instrumentation import (
    instrument_model, deinstrument_model, get_model_type, load_model,
)


def tiny_model(attention, projection="relu"):
    return LongT5ForConditionalGeneration(LongT5Config(
        vocab_size=32, d_model=16, d_kv=4, d_ff=32, num_layers=2,
        num_decoder_layers=1, num_heads=2, local_radius=2, global_block_size=4,
        encoder_attention_type=attention, feed_forward_proj=projection,
        dropout_rate=0, decoder_start_token_id=0, eos_token_id=1, pad_token_id=0,
    )).eval()


@pytest.mark.parametrize("attention", ["local", "transient-global"])
@pytest.mark.parametrize("projection", ["relu", "gated-gelu"])
def test_longt5_collects_real_activations_without_changing_generation(attention, projection):
    torch.manual_seed(7)
    model = tiny_model(attention, projection)
    inputs = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 0, 0, 0, 0]])
    mask = inputs.ne(0).long()
    with torch.no_grad():
        expected = model.generate(inputs, attention_mask=mask, max_new_tokens=3)
    collector = Collector({"zone_size_by_layer": {"default": 1}, "layer_names_to_include": ["all"], "layer_indexes_to_include": []})
    config = dict(instrument_encoder=True, instrument_decoder=True, instrument_FF=True, instrument_attention=True)
    handles = instrument_model(config, model, collector)
    try:
        with torch.no_grad():
            actual = model.generate(inputs, attention_mask=mask, max_new_tokens=3)
        assert torch.equal(expected, actual)
        names = collector.layer_id_to_name
        assert any("encoder.LongT5Block.1" in name for name in names.values())
        assert any("DenseReluDense.wo" in name for name in names.values())
        assert any("EncDecAttention.q" in name for name in names.values())
        assert "decoder.output" in names.values()
        if attention == "transient-global":
            token = next(key for key, name in names.items() if name.endswith("TransientGlobalSelfAttention.k"))
            glob = next(key for key, name in names.items() if name.endswith("TransientGlobalSelfAttention.global_k"))
            assert collector.outputs[0][token].shape[0] == 8
            assert collector.outputs[0][glob].shape[0] == 2
        collector.finish_and_get_data()
    finally:
        deinstrument_model(handles)
    assert handles == []
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("encoder,decoder,ff,attention", [(True,False,True,False),(False,True,False,True),(False,False,False,False)])
def test_longt5_respects_selection_flags(encoder, decoder, ff, attention):
    model = tiny_model("transient-global")
    collector = Collector({"zone_size_by_layer": {"default": 1}, "layer_names_to_include": ["all"], "layer_indexes_to_include": []})
    handles = instrument_model(dict(instrument_encoder=encoder,instrument_decoder=decoder,instrument_FF=ff,instrument_attention=attention),model,collector)
    names = [m.ns_name for m in model.modules() if m._forward_hooks]
    assert any(n.startswith("encoder.") for n in names) == encoder
    assert any("DenseReluDense" in n for n in names) == ff
    assert any("Attention" in n for n in names) == attention
    assert "decoder.output" in names
    deinstrument_model(handles)


@pytest.mark.parametrize("name", ["google/long-t5-tglobal-large", "google/long-t5-local-base", "local/longt5-checkpoint"])
@pytest.mark.parametrize("quantization", ["none", "int8", "int4"])
def test_longt5_loads_as_seq2seq(monkeypatch, name, quantization):
    import neuralsignal.core.modules.model_instrumentation as module
    calls = []
    monkeypatch.setattr(module, "BitsAndBytesConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(module.AutoModelForSeq2SeqLM, "from_pretrained", lambda *args, **kwargs: calls.append((args, kwargs)) or "model")
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *args: "tokenizer")
    assert get_model_type(name) == "longt5"
    assert load_model(dict(model_name=name, device="cpu", quantization=quantization)) == ("tokenizer", "model")
    assert len(calls) == 1
    assert ("quantization_config" in calls[0][1]) == (quantization != "none")
