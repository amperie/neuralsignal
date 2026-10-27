import pytest
import torch
from torch import nn

from neuralsignal.core.modules.model_instrumentation import check_embeddings_nonzero, tie_seq2seq_embeddings
from neuralsignal.features.model_extractor import FeatureHealthError, ModelFeatureExtractor, check_feature_health


class Stack(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.embed_tokens = emb


class Seq2Seq(nn.Module):
    def __init__(self, zero_stacks=True):
        super().__init__()
        self.shared = nn.Embedding(10, 4)
        make = (lambda: nn.Embedding(10, 4)) if not zero_stacks else (lambda: _zero(nn.Embedding(10, 4)))
        self.encoder = Stack(make())
        self.decoder = Stack(make())


def _zero(m):
    nn.init.zeros_(m.weight)
    return m


def test_tie_fixes_zero_stack_embeddings():
    m = Seq2Seq()
    with pytest.raises(RuntimeError, match="all zeros"):
        check_embeddings_nonzero(m, "x")
    tie_seq2seq_embeddings(m)
    assert m.encoder.embed_tokens is m.shared and m.decoder.embed_tokens is m.shared
    check_embeddings_nonzero(m, "x")


def test_already_tied_model_unchanged():
    m = Seq2Seq(zero_stacks=False)
    m.encoder.embed_tokens = m.shared
    m.decoder.embed_tokens = m.shared
    tie_seq2seq_embeddings(m)
    assert m.encoder.embed_tokens is m.shared
    check_embeddings_nonzero(m)


def test_zero_shared_raises():
    m = Seq2Seq()
    _zero(m.shared)
    tie_seq2seq_embeddings(m)
    with pytest.raises(RuntimeError, match="shared"):
        check_embeddings_nonzero(m)


def test_health_passes_on_varied_rows():
    rows = [{"a": float(i), "b": float(i * 2), "c": 1.0} for i in range(5)]
    assert check_feature_health(rows, ["yes"] * 5)["constant_fraction"] == pytest.approx(1 / 3)


def test_health_fails_on_mostly_constant():
    rows = [{"a": float(i), "b": 0.0, "c": 0.0} for i in range(5)]
    with pytest.raises(FeatureHealthError, match="constant"):
        check_feature_health(rows, ["x"] * 5)


def test_health_fails_on_empty_generation():
    rows = [{"a": float(i)} for i in range(5)]
    with pytest.raises(FeatureHealthError, match="empty"):
        check_feature_health(rows, ["", " "] * 2)


def test_extractor_runs_check_once_after_threshold(monkeypatch):
    calls = []
    import neuralsignal.features.model_extractor as me
    monkeypatch.setattr(me, "check_feature_health", lambda r, t, f: calls.append(len(r)))
    ex = ModelFeatureExtractor({"health_check": {"rows": 3}})
    assert not ex._hc_done and ex._hc_rows == 3
    ex2 = ModelFeatureExtractor({"health_check": {"enabled": False}})
    assert ex2._hc_done
