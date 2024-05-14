import yaml
import os
import logging
# from neuralsignal.backend.ns_backend import NSBackend


class NeuralSignalConfig:
    """
    Configuration class for the NeuralSignal SDK
    Used as a semi-static class. Usage:

    from neuralsignal.core.modules.neuralsignal_config import sdk_config
    sdk_config.get(...)
    """

    def __init__(self, config_file: str = None) -> None:
        if config_file is None:
            config_file = "neuralsignal/sdk/neuralsignal_sdk.yaml"
        self.config_file = config_file
        self._initialize(config_file)

    def _initialize(self, config_file: str = None) -> None:
        self.config = yaml.safe_load(
            open(config_file))

    def get_config(self) -> dict:
        return self.config

    def set_config(self, config: dict) -> None:
        self.config = config

    def get(self, key: str) -> any:
        return self.config[key]

    def set(self, key: str, value: any) -> None:
        self.config[key] = value

    def refresh(self, config_file: str = None) -> None:
        self._initialize(config_file)

    def get_backend_config(self) -> dict:
        return self.config["backend_config"]

    def get_detector_configs(self) -> dict:
        return self.config["detectors"]

    def get_detector_config(self, detector_name: str) -> dict:
        ds = self.get_detector_configs()
        for d in ds:
            if d['behavior_name'] == detector_name:
                return d
        raise ValueError(f"Could not find detector {detector_name}")

    def get_sdk_home(self) -> str:
        return self.get("home")

    def logging_level(self):
        ll = self.get("logging_level").upper()
        if ll == "DEBUG":
            return logging.DEBUG
        if ll == "INFO":
            return logging.INFO
        if ll == "WARN":
            return logging.WARN
        if ll == "ERROR":
            return logging.ERROR
        if ll == "CRITICAL":
            return logging.CRITICAL
        # If none of the above, return default INFO
        return logging.INFO


try:
    sdk_config
except NameError:
    sdk_config = None

if sdk_config is None:
    sdk_config = NeuralSignalConfig()

# Make sure home directory exists
if not os.path.exists(sdk_config.get("home")):
    os.mkdir(sdk_config.get("home"), 0o777)
if not os.path.exists(f'{sdk_config.get("home")}/s1'):
    os.mkdir(f'{sdk_config.get("home")}/s1', 0o777)

# TODO: Maybe put some logic in here to clean up caches
# for items that haven't been used in a while
