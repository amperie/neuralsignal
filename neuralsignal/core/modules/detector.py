import logging
from neuralsignal.core.modules.tensors import featurize_tensor_dict
from neuralsignal.core.modules.utils import generate_uuid
# from neuralsignal.backend.ns_backend import NSBackend

logging.basicConfig(level=logging.INFO)

"""
Wraps an S1 model plus all the settings needed for real time evaluation:
- S1 model
- Prompt
- Metadata/Name of the behavior it's detecting
"""


class DetectionResults:
    """Results of a detection
    """
    def __init__(self, behavior_name: str, score: float):
        self.behavior_name = behavior_name
        self.score = score
        self.threshold = None
        self.correlation_id = generate_uuid()

    def __str__(self):
        return f"DetectionResults: {self.behavior_name} - {self.score}"

    def get_data(self) -> dict:
        return {
            "behavior_name": self.behavior_name,
            "score": self.score,
            "threshold": self.threshold,
            "correlation_id": self.correlation_id
        }


class Detector:
    """Class for attaching detectors to evaluation methods
    """
    default_config = {
        "S1_model": None,  # Either pass the model directly or specify its path
        "S1_model_path": None,  # If both are present S1_model is used
        "prompt": "",
        "behavior_name": "default",
        "threshold": None,
        "enabled": False,
    }

    def __init__(self, config: dict = None, be=None) -> None:
        if config is None:
            config = self.default_config
        else:
            self.config = {**self.default_config, **config}
        logging.debug(f"Initializing detector with config: {self.config}")
        if self.config["S1_model"] is not None:
            self.model = self.config["S1_model"]
        elif self.config["S1_model_path"] is not None:
            if be is None:
                raise ValueError(
                    "Backend must be provided if S1_model_path is used")
            self.model = be.load_s1_model(self.config["S1_model_path"])
        else:
            raise ValueError("S1_model or S1_model_path must be provided")
        self.enabled = self.config["enabled"]
        self.behavior_name = self.config["behavior_name"]
        self.prompt = self.config["prompt"]

    def predict(self, input_data: list) -> float:
        """Runs the S1 model and returns the probability of
        class 0 being detected

        Args:
            input_data (list): Featurized list of zones

        Returns:
            float: probability of class 0 being detected
        """
        try:
            # return self.model.predict_proba([input_data])
            # for testing
            if self.behavior_name == "toxicity":
                return .1
            return .5
        except ValueError as e:
            logging.error(
                f"Error predicting for detector: {self.behavior_name}\n"
                "Disabling detector\n"
                f"Error: {e}")
            self.enabled = False
            return None

    def detect(self, input_data) -> DetectionResults:
        """Detects behavior in input data
        If a threshold is defined, returns a binary value of 0 or 1
        If a threshold is not defined, returns a probability between 0 and 1
        """
        if not self.enabled:
            return None
        fd = featurize_tensor_dict(input_data, 1, 1)
        prob_class_0 = self.predict(fd[0])
        retVal = DetectionResults(self.config["behavior_name"], prob_class_0)
        return retVal
