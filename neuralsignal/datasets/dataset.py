from datasets import load_dataset
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
        "gt_processor": None,
        "input_processor": None,
        "column_map": {
            "input_column": "input",
            "gt_column": "ground_truth"
        },
        "prompt": "",
        "split": "train+test+validation",
        "config_name": None,
        "row_limit": 0,
        "starting_row": 0,
        "truncate_input": 0,
        "truncation_start": 0,
    }

    def __init__(self, config) -> None:
        """wrapper for HF datasets. Config should contain:
        dataset_name: str, name of the dataset
        hf_token: str, huggingface token
        column_map: dict, maps the names of the dataset columns 
            to the names the code expects. Should contain:
            input_column: str, name of the input column
            gt_column: str, name of the ground truth column
            row_limit: limit of how many rows. 0 for no limit
        gt_processor: function, optional. funciton to change
            the ground truth to what is expected by the model.
            Input is a string, output is a string. 
            ie: change 1 -> positive and 0 -> negative
        input_processor: function, optional. function to process
            the input. The argument into it is the dataset itself
            The return value should be the list of inputs
        prompt: str, optional. Prompt to prepend to the input
        config_name: dataset HF config
        row_limit: how many rows to load. 0 for no limit
        starting_row: int, row to start from
        split: str, HF split(s) to load
        truncate_input: limit length of input string
            0 -> don't truncate

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
            case "hf_exploded":
                self.load_hf_exploded_dataset()
            case _:
                raise ValueError(
                    "Dataset type "
                    f"{self.config['dataset_type']} not supported")

    def load_local_json_one_per_line_dataset(self):
        f = open(self.config["local_path"], "r", encoding="utf8")
        self.dataset = []
        self.rows = []
        self.line_number = []
        i = 1
        for line in f:
            if i >= self.config["starting_row"]:
                row = json.loads(line)
                self.dataset.append(row)

                row_dict = self.config["input_processor"](row)

                self.rows.append(row_dict)
                self.line_number.append(i)

            i += 1
            if i - self.config["starting_row"] > self.config["row_limit"]\
                    and self.config["row_limit"] > 0:
                break

        self.loaded = True

    def load_hf_exploded_dataset(self):
        raise NotImplementedError(
            "load_hf_exploded_dataset not implemented")

    def load_hf_dataset(self):
        raise NotImplementedError(
            "load_hf_dataset not implemented")

    def __iter__(self):
        """returns a tuple with prompt+input and ground truth

        Yields:
            Tuple(str,str): Tuple of input and ground truth
        """
        for row, line_number in zip(
                self.rows, self.line_number):
            yield row, line_number
