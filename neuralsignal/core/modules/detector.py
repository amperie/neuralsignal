import logging
from neuralsignal.core.modules.s1_models import load_model_from_mlflow

logging.basicConfig(level=logging.INFO)

"""
Wraps an S1 model plus all the settings needed for real time evaluation:
- S1 model
- Prompt
- Metadata/Name of the behavior it's detecting
"""


class Detector:
    """Class for attaching detectors to evaluation methods
    """
    default_config = {
        "S1_model": None,  # Either pass the model directly or specify its path
        "S1_model_path": None,  # If both are present S1_model is used
        "mlflow_uri": None,
        "prompt": "",
        "behavior_name": "default",
        "threshold": None,
    }

    def __init__(self, config: dict = None):
        if config is None:
            config = self.default_config
        else:
            self.config = {**self.default_config, **config}
        logging.debug(f"Initializing detector with config: {self.config}")
        if self.config["S1_model"] is not None:
            self.model = self.config["S1_model"]
        elif self.config["S1_model_path"] is not None:
            if self.config["mlflow_uri"] is None:
                raise ValueError(
                    "mlflow_uri must be provided if S1_model_path is used")
            self.model = load_model_from_mlflow(
                self.config["mlflow_uri"], self.config["S1_model_path"])
        else:
            raise ValueError("S1_model or S1_model_path must be provided")

    def detect(self, input_data) -> float:
        """Detects behavior in input data
        If a threshold is defined, returns a binary value of 0 or 1
        If a threshold is not defined, returns a probability between 0 and 1
        """
        pass
