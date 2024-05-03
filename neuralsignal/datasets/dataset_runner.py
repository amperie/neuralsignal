from neuralsignal.sdk.neuralsignal import SDK
from neuralsignal.datasets.dataset_definitions import get_dataset


class DatasetRunner:
    """Loads a dataset and runs data collection for it
    """
    default_config = {
            "dataset": None,
            "row_limit": 0,
            "batch_size": 1,
            "detectors": [],
        }

    def _initialize(self, dataset_name, detectors: list):
        """Initializes the dataset and the SDK
        """
        self.sdk = SDK()
        self.sdk.set_config("save_scans", True)
        self.dataset = get_dataset(dataset_name)
        self.dataset.load()
        self.detectors = detectors

    def __init__(self, config: dict):
        """
        Config should contain general configs:
        {
            "dataset": "name",
            "row_limit": 0,
            "batch_size": 1,
        }
        """
        config = {**self.default_config, **config}
        self.config = config
        self.batch_size = config["batch_size"]
        if config["dataset"] is None:
            raise ValueError("Dataset name must be provided")
        if len(config["detectors"]) == 0:
            raise ValueError("Detectors must be provided")
        self._initialize(config["dataset"], config["detectors"])

    def run(self):
        """Runs the data collection for all required rows in the dataset
        """
        batch = []

        # Batch up the inputs
        for row in self.dataset:
            batch.append(row)
            if len(batch) == self.batch_size:
                self.sdk.evaluate_indirect_output(batch, self.detectors)
                batch = []
        # Catch the last batch
        if len(batch) > 0:
            self.sdk.evaluate_indirect_output(batch, self.detectors)
