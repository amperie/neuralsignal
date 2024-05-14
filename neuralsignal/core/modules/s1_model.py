import logging
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


class S1Model:
    """
    Simple class to just wrap a model so we can include metadata
    about it an define interfaces for using it
    """

    default_config = {
        "application_name": None,
        "sub_application_name": None,
        "model_name": None,
        "model_id": None,
        "dataset_path": None,
        "optimization_metric": None,
        'metrics': {},
        'params': {},
        'metadata': {},
        'tags': {},
        'artifacts': {},
        'description': "",
    }

    def __init__(self, config: dict) -> None:

        self.config = {**self.default_config, **config}
        self.model_name = config["model_name"]
        self.model_id = config["model_id"]
        self.model = config['model']

    def set_id(self, model_id: str):
        self.model_id = model_id

    def __getitem__(self, name: str):
        return self.config[name]

    def predict(self, data):
        pass

    def predict_proba(self, data):
        pass
