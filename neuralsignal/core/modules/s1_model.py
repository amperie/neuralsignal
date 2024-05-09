import logging
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


class S1Model:
    """
    Simple class to just wrap a model so we can include metadata
    about it an define interfaces for using it
    """
    def __init__(self, config: dict) -> None:
        self.config = config
        self.name = config["name"]
        self.model_id = config["model_id"]
        self.model = config['model']


