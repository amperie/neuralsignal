import logging
import pickle
import os
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


class NSModel:

    def __init__(self, config: dict) -> None:
        self.config = config
        self.name = config["name"]
        self.model = config["model"]

    def add_config(self, config: dict):
        self.config = {**self.config, **config}

    def get_name(self):
        return self.name

    def get_model(self):
        return self.model

    def predict(self, data):
        return self.model(data)

    def save_to_disk(self):
        path = f"{sdk_config.get_sdk_home()}/models/"
        os.makedirs(path, exist_ok=True)
        path = f"{path}{self.name}.NSModel"
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(name: str):
        path = f"{sdk_config.get_sdk_home()}/models/{name}.NSModel"
        with open(path, 'rb') as f:
            return pickle.load(f)
