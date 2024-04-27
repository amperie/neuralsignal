import yaml
import os
# from neuralsignal.backend.ns_backend import NSBackend


class NeuralSignalConfig:
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
