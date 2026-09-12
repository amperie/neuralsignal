import pytest

from neuralsignal.remote.lifecycle import _available_gpus, _prompt_gpu_choice, _with_selected_gpu
from neuralsignal.remote.runpod import RunPodGpuType


def gpu(name, memory, price, stock="High", counts=(1,)):
    return RunPodGpuType(name, name, memory, stock, price, counts)


def test_filter_includes_boundaries_and_sorts_by_price():
    choices = _available_gpus([
        gpu("lower", 18, 0.8), gpu("upper", 30, 0.2),
        gpu("small", 17, 0.1), gpu("large", 31, 0.1),
        gpu("unavailable", 24, 0.1, "None"),
        gpu("wrong-count", 24, 0.1, counts=(2,)),
        gpu("unknown-price", 24, None),
    ], 24, 1)
    assert [g.id for g in choices] == ["upper", "lower", "unknown-price"]


def test_yes_selects_cheapest_without_input(monkeypatch, capsys):
    def unexpected(*args):
        pytest.fail("must not prompt")
    monkeypatch.setattr("builtins.input", unexpected)
    choices = [gpu("expensive", 18, 0.8), gpu("cheap", 30, 0.2), gpu("unknown", 24, None)]
    assert _prompt_gpu_choice(choices, yes=True).id == "cheap"
    assert "$0.200/hr" in capsys.readouterr().out


def test_yes_requires_known_price():
    with pytest.raises(RuntimeError, match="known price"):
        _prompt_gpu_choice([gpu("unknown", 24, None)], yes=True)


@pytest.mark.parametrize("answer", ["0", "-1", "3", "invalid"])
def test_invalid_selection_is_rejected(monkeypatch, answer):
    monkeypatch.setattr("builtins.input", lambda _: answer)
    with pytest.raises(RuntimeError, match="Invalid GPU selection"):
        _prompt_gpu_choice([gpu("one", 24, 0.5), gpu("two", 24, 0.6)])


def test_interactive_selection(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert _prompt_gpu_choice([gpu("one", 24, 0.5), gpu("two", 24, 0.6)]).id == "two"


def test_yes_queries_and_selects_for_manifest():
    class Api:
        def list_gpu_types(self, count, secure):
            assert (count, secure) == (1, True)
            return [gpu("one", 18, 0.9), gpu("two", 24, 0.3)]
    result = _with_selected_gpu({"runpod": {}}, Api(), 24, None, yes=True)
    assert result["runpod"]["gpu_type_ids"] == ["two"]


def test_cli_gb_overrides_yaml_gpu_and_forwards_yes(tmp_path, monkeypatch):
    import importlib
    cli = importlib.import_module("neuralsignal.cli.main")
    config = tmp_path / "launch.yaml"
    config.write_text("remote_collect:\n  config: features.yaml\n  run_id: test\n  gpu_id: old-gpu\n")
    captured = {}
    def collect(*args, **kwargs):
        captured.update(kwargs)
        return {}
    monkeypatch.setattr(cli, "remote_collect_lifecycle", collect)
    cli.main(["remote", "collect", str(config), "-gb", "24", "--yes"])
    assert captured["gpu_vram_gb"] == 24
    assert captured["gpu_id"] is None
    assert captured["yes"] is True
