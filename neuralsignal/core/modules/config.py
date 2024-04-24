import yaml


class Config:
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
