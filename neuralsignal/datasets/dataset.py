import logging
import json

logging.basicConfig(level=logging.INFO)


# Example gt_processor:
def change_gt(val):
    if val == -1:
        return "negative"
    if val == 0:
        return "neutral"
    return "positive"


class NSDataset:

    # dataset_type: can be hf, local_json_one_per_line
    default_config = {
        "dataset_type": "hf",
        "local_path": None,
        "dataset_variant": None,
        "input_processor": None,
        "split": "train+test+validation",
        "config_name": None,
        "row_limit": 0,
        "starting_row": 0,
    }

    def __init__(self, config) -> None:
        """wrapper for HF datasets. Config should contain:
        dataset_name: str, name of the dataset
        hf_token: str, huggingface token
        input_processor: function that takes a row and returns a dict:
            input: input to the model (user's query)
            context: any context sent in with the input
            output: output of the model that is being evaluated
            ground_truth: if there is one
            metadata: dict of any fields that will pass through

        config_name: dataset HF config
        row_limit: how many rows to load. 0 for no limit
        starting_row: int, row to start from
        split: str, HF split(s) to load

        Args:
            config (dict): config dictionary
        """
        # Merge with the default config in case things are missing
        self.config = {**self.default_config, **config}
        self.loaded = False

    def load(self):
        """Loads the dataset into memory"""
        logging.info(f"Loading dataset {self.config['dataset_name']}")
        if self.loaded:
            return

        match self.config["dataset_type"]:
            case "hf":
                self.load_hf_dataset()
            case "local_json_one_per_line":
                self.load_local_json_one_per_line_dataset()
            case _:
                raise ValueError(
                    "Dataset type "
                    f"{self.config['dataset_type']} not supported")

    def load_local_json_one_per_line_dataset(self):
        f = open(self.config["local_path"], "r", encoding="utf8")
        self.rows = []
        self.line_number = []
        i = 1
        for line in f:
            if i >= self.config["starting_row"]:
                row = json.loads(line)

                row_dict = self.config["input_processor"](row)
                row_dict["line_number"] = i

                self.rows.append(row_dict)

            i += 1
            if i - self.config["starting_row"] > self.config["row_limit"]\
                    and self.config["row_limit"] > 0:
                break

        self.loaded = True

    def load_hf_dataset(self):
        raise NotImplementedError(
            "load_hf_dataset not implemented")

    def __iter__(self):
        """returns a tuple with prompt+input and ground truth

        Yields:
            Tuple(str,str): Tuple of input and ground truth
        """
        # Make sure the dataset is loaded first
        # This should only run once with the yield statement
        if not self.loaded:
            self.load()

        for row in self.rows:
            yield row
