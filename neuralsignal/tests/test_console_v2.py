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


def test_provider_noise_filter_preserves_warnings_and_application_progress():
    from neuralsignal.console import ProviderNoiseFilter, QUIET_LOGGERS

    noise_filter = ProviderNoiseFilter()
    for provider in QUIET_LOGGERS:
        for name in (provider, provider + ".client"):
            for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
                record = logging.LogRecord(name, level, "", 1, "message", (), None)
                assert noise_filter.filter(record) == (level >= logging.WARNING)
    for name in ("neuralsignal.remote.job", "__main__", "datasets_custom"):
        record = logging.LogRecord(name, logging.INFO, "", 1, "progress", (), None)
        assert noise_filter.filter(record)


def test_configured_handler_suppresses_http_urls_even_with_explicit_child_level(monkeypatch):
    from neuralsignal.console import configure_logging, QUIET_LOGGERS

    stream = io.StringIO()
    monkeypatch.setattr("sys.stderr", stream)
    monkeypatch.setenv("NEURALSIGNAL_LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    levels = {name: logging.getLogger(name).level for name in (*QUIET_LOGGERS, "httpx.client")}
    try:
        configure_logging()
        child = logging.getLogger("httpx.client")
        child.setLevel(logging.DEBUG)
        child.info("GET signed-url-that-should-not-appear")
        child.warning("Connection retry needed")
        logging.getLogger("neuralsignal.remote.job").info("Extracting batch")
        output = stream.getvalue()
        assert "signed-url" not in output
        assert "Connection retry needed" in output
        assert "Extracting batch" in output
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers = original_handlers
        root.setLevel(original_level)
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
