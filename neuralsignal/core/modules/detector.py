import logging
import copy
from neuralsignal.core.modules.tensors import featurize_tensor_dict
from neuralsignal.core.modules.utils import generate_uuid
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.backend.ns_backend import NSBackend

logging.basicConfig(level=logging.INFO)

"""
Wraps an S1 model plus all the settings needed for real time evaluation:
- S1 model
TODO: S1 Model should have been trained on the same prompt. 
So need to track this as well
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
        self.prompted_input = None

    def __str__(self):
        return f"DetectionResults: {self.behavior_name} - {self.score}"

    def get_data(self) -> dict:
        return {
            "behavior_name": self.behavior_name,
            "score": self.score,
            "threshold": self.threshold,
            "correlation_id": self.correlation_id,
            "prompted_input": self.prompted_input,
        }


class Detector:
    """Class for attaching detectors to evaluation methods
    """
    default_config = {
        "S1_model": None,  # Either pass the model directly or specify its path
        "S1_model_path": None,  # If both are present S1_model is used
        "prompt": "",
        "input_prompt": "",
        "behavior_name": "default",
        "threshold": None,
        "enabled": False,
    }

    def __init__(self, config: dict = None) -> None:
        """Initialize a detector

        Args:
            config (dict, optional): Should contain these setting:
            S1_model: S1Model object of a loaded model 
            OR:
            S1_model_path (str, optional): URI of the model
            prompt: detector specific prompt for indirect mode
            behavior_name: name of the behavior to detect
            threshold: threshold for the detector
            enabled: whether the detector is enabled
            application_name: name of the application
            sub_application_name: name of the sub application

            Defaults to None, in which case the default config
            from the config file is used

        Raises:
            ValueError: If application_name of sub_application name is missing
            ValueError: If S1 model or model path is not provided
        """
        if "application_name" not in config:
            raise ValueError("Missing application_name in config")
        if "sub_application_name" not in config:
            raise ValueError("Missing sub_application_name in config")
        if config is None:
            config = sdk_config.get_config()
        else:
            self.config = {**self.default_config, **config}
        logging.debug(f"Initializing detector with config: {self.config}")
        self.backend = copy.deepcopy(
            sdk_config.get_backend_config())

        self.backend["application_name"] =\
            self.config["application_name"]
        self.backend["sub_application_name"] =\
            self.config["sub_application_name"]
        self.backend = NSBackend(self.backend)
        if self.config["S1_model"] is not None:
            self.model = self.config["S1_model"]
            logging.info(f"Loaded S1 model directly: {self.model}")
        elif self.config["S1_model_path"] is not None:
            self.model =\
                self.backend.load_s1_model(self.config["S1_model_path"])
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
