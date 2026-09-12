import io
import logging

from neuralsignal.console import ConsoleFormatter, color
from neuralsignal.remote.lifecycle import _prompt_gpu_choice
from neuralsignal.remote.runpod import RunPodGpuType


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_color_only_on_terminal_and_respects_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    assert color("model", "cyan", Terminal()) == "\033[36mmodel\033[0m"
    assert color("model", "cyan", io.StringIO()) == "model"
    monkeypatch.setenv("NO_COLOR", "")
    assert color("model", "cyan", Terminal()) == "model"


def test_log_formatter_colors_metadata_without_changing_record(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    record = logging.LogRecord("worker", logging.WARNING, "", 1, "waiting %s", ("for GPU",), None)
    output = ConsoleFormatter(Terminal()).format(record)
    assert "\033[33mWARNING" in output
    assert "\033[2mworker" in output
    assert output.endswith("waiting for GPU")
    assert record.levelname == "WARNING" and record.name == "worker"
    assert "\033[" not in ConsoleFormatter(io.StringIO()).format(record)


def test_gpu_model_and_price_are_colored(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    stream = Terminal()
    monkeypatch.setattr("sys.stdout", stream)
    _prompt_gpu_choice([RunPodGpuType("gpu", "RTX 4090", 24, "High", 0.3, (1,))], yes=True)
    assert "\033[36mRTX 4090\033[0m" in stream.getvalue()
    assert "\033[32m$0.300/hr\033[0m" in stream.getvalue()
