import logging
from neuralsignal.core.modules.detector import DetectionResults

logging.basicConfig(level=logging.INFO)


class GenerationInstance:
    """Represents a single instance of a generation task
    and its associated data and metada
    """

    default_config = {
        "ground_truth": None,
        "input": None,
        "output": None,
        "data_run_name": None,
        "model_name": None,
    }

    def __init__(self, config: dict = None) -> None:
        """Initializes a GenerationInstance

        Args:
            config (dict): Dictionary of configuration options
        """
        if config is None:
            config = self.default_config
        # Merge the config with the default config in case
        # the config is incomplete
        self.config = {**self.default_config, **config}
        self.data = {
            "ground_truth": self.config["ground_truth"],
            "input": self.config["input"],
            "output": self.config["output"],
            "data_run_name": self.config["data_run_name"],
            "model_name": self.config["model_name"],
        }
        self.detections = {}
        self.data_to_save = {}

    def add_data(self, data: dict) -> None:
        """Adds data to the instance

        Args:
            data (dict): Dictionary of data to add
        """
        self.data = {**self.data, **data}

    def add_data_to_save(self, data: dict) -> None:
        """Adds data to the instance

        Args:
            data (dict): Dictionary of data to add
        """
        self.data = {**self.data, **data}

    def get_flattened_data(self) -> dict:
        """Returns a flattened version of the data

        Returns:
            dict: Flattened version of the data
        """
        # TODO: redo this since some of these fields may not be there
        flat_data = {
            "ground_truth": self.data["ground_truth"],
            "input": self.data["input"],
            "output": self.data["output"],
            "data_run_name": self.data["data_run_name"],
            "model_name": self.data["model_name"],
            "context": self.data["context"],
            "metadata": self.data["metadata"],
            "zone_size": self.data["zone_size"],
            "layer_names": self.data["layer_names"],
            "layer_order": self.data["layer_order"],
            "layer_id_to_name": self.data["layer_id_to_name"],
            "layer_passes_by_name": self.data["layer_passes_by_name"],
            "layer_passes": self.data["layer_passes"],
            "topology": self.data["topology"],
        }
        return {**flat_data, **self.data_to_save}

    def add_detection(self, detection: DetectionResults):
        """Adds detections to the instance

        Args:
            detections (DetectionResults): Detection Results to add
        """
        self.detections[detection.behavior_name] = detection
