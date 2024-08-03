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
        """Adds  arbitrary data to the instance

        Args:
            data (dict): Dictionary of data to add
        """
        self.data_to_save = {**self.data_to_save, **data}

    def get_flattened_data(self) -> dict:
        """Returns a flattened version of the data

        Returns:
            dict: Flattened version of the data
        """
        # TODO: redo this since some of these fields may not be there
        flat_data = {}
        if "ground_truth" in self.data:
            flat_data["ground_truth"] = self.data["ground_truth"]
        if "input" in self.data:
            flat_data["input"] = self.data["input"]
        if "output" in self.data:
            flat_data["output"] = self.data["output"]
        if "zone_sizes_by_layer" in self.data:
            flat_data["zone_sizes_by_layer"] =\
                self.data["zone_sizes_by_layer"]
        if "decoded_output" in self.data:
            flat_data["decoded_output"] = self.data["decoded_output"]
        if "data_run_name" in self.data:
            flat_data["data_run_name"] = self.data["data_run_name"]
        if "model_name" in self.data:
            flat_data["model_name"] = self.data["model_name"]
        if "context" in self.data:
            flat_data["context"] = self.data["context"]
        if "metadata" in self.data:
            flat_data["metadata"] = self.data["metadata"]
        if "zone_size" in self.data:
            flat_data["zone_size"] = self.data["zone_size"]
        if "layer_names" in self.data:
            flat_data["layer_names"] = self.data["layer_names"]
        if "layer_order" in self.data:
            flat_data["layer_order"] = self.data["layer_order"]
        if "layer_id_to_name" in self.data:
            flat_data["layer_id_to_name"] = self.data["layer_id_to_name"]
        if "layer_passes_by_name" in self.data:
            flat_data["layer_passes_by_name"] =\
                self.data["layer_passes_by_name"]
        if "layer_passes" in self.data:
            flat_data["layer_passes"] = self.data["layer_passes"]
        if "topology" in self.data:
            flat_data["topology"] = self.data["topology"]
        if "outputs" in self.data:
            flat_data["outputs"] = self.data["outputs"]
        if "inputs" in self.data:
            flat_data["inputs"] = self.data["inputs"]
        if "generation_correlation_id" in self.data:
            flat_data["generation_correlation_id"] =\
                self.data["generation_correlation_id"]
        detections = {}
        for d in self.detections.keys():
            detections[d] = self.detections[d].get_data()
        flat_data["detections"] = detections
        return {**flat_data, **self.data_to_save}

    def add_detection(self, detection: DetectionResults):
        """Adds detections to the instance

        Args:
            detections (DetectionResults): Detection Results to add
        """
        self.detections[detection.behavior_name] = detection
