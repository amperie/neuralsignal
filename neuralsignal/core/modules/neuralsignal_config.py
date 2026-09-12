import logging


class NeuralSignalConfig:
    def __init__(self, config: dict | None = None) -> None:
        self.config = {"logging_level": "INFO", "home": "sdk_home", **(config or {})}

    def get_config(self) -> dict:
        return self.config

    def set_config(self, config: dict) -> None:
        self.config = config

    def get(self, key: str):
        return self.config[key]

    def set(self, key: str, value) -> None:
        self.config[key] = value

    def refresh(self, config_file: str | None = None) -> None:
        if config_file is not None:
            raise RuntimeError("File-backed sdk_config was removed with the v1 SDK path")

    def get_backend_config(self) -> dict:
        raise RuntimeError("The v1 backend config was removed")

    def get_detector_configs(self) -> list[dict]:
        return list(self.config.get("detectors", []))

    def get_detector_config(self, detector_name: str) -> dict:
        for detector in self.get_detector_configs():
            if detector.get("behavior_name") == detector_name:
                return detector
        raise ValueError(f"Could not find detector {detector_name}")

    def get_sdk_home(self) -> str:
        return str(self.config.get("home", "sdk_home"))

    def logging_level(self):
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARN,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return levels.get(str(self.config.get("logging_level", "INFO")).upper(), logging.INFO)


sdk_config = NeuralSignalConfig()