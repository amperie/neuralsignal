import logging
from neuralsignal.sdk.neuralsignal import SDK
from neuralsignal.datasets.dataset_definitions import get_dataset
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


class DatasetRunner:
    """Loads a dataset and runs data collection for it
    """
    default_config = {
            "dataset": None,
            "row_limit": 0,
            "batch_size": 1,
            "detectors": [],
            "max_new_tokens": 128,
        }

    def _initialize(self, config: dict):
        """Initializes the dataset and the SDK
        """
        self.sdk = SDK(
            config['application_name'],
            config['sub_application_name'],
            config=config
            )
        self.sdk.set_config("save_scans", True)
        self.sdk.set_config("max_new_tokens", config["max_new_tokens"])
        self.dataset = get_dataset(
            row_limit=config['row_limit'],
            dataset_name=config['dataset'])
        self.dataset.load()
        self.detectors = config['detectors']

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
        self._initialize(config)

    def run(self):
        """Runs the data collection for all required rows in the dataset
        """
        batch = []
        index = 1

        # Batch up the inputs
        for row in self.dataset:
            logging.debug(
                f"Batching row {index} for "
                f"batch size {self.batch_size}")
            batch.append(row)
            index += 1
            if len(batch) == self.batch_size:
                logging.debug(
                    f"Evaluating batch from rows "
                    f"{index - self.batch_size + 1} to {index}")
                self.sdk.evaluate_indirect_output(batch, self.detectors)
                batch = []
                logging.debug(
                    f"Done evaluating batch from rows "
                    f"{index - self.batch_size + 1} to {index}")
        # Catch the last batch
        if len(batch) > 0:
            logging.debug(
                f"Evaluating last batch from rows "
                f"{index - self.batch_size + 1} to {index}")
            self.sdk.evaluate_indirect_output(batch, self.detectors)
            logging.debug(
                f"Done evaluating last batch from rows "
                f"{index - self.batch_size + 1} to {index}")
