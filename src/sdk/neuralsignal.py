import logging
import json
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

logging.basicConfig(level=logging.INFO)


class sdk:
    """Main entrypoint into NeuralSignal SDK
    """

    default_config = {
        "evaluation_mode": "qb",  # qb or direct_instrument
        "qb_config": {
            "qb_model": "google/flan-t5-large",
            "zone_size:": 1024,
            "qb_batch_size": 1,
            "quantization": "int8",
            "device": "cuda",
        },
        "save_scans": False,  # Save scans to backend
        "backend_config": {},  # Backend endpoint
        "S1_model": None,  # S1 model can't be None
    }

    def __init_qb():
        pass

    def __init__(self, config: dict) -> None:
        json_str = json.dumps(config, indent=4, sort_keys=False)
        log_string = highlight(json_str, JsonLexer(), TerminalFormatter())
        logging.info(log_string)

        self.cfg = config
        if config["evaluation_mode"] == "qb":
            self.__init_qb()

    def evaluate_single_output(self, output: dict) -> dict:
        """Evaluates a single output

        Args:
            output (dict): dictionary that contains the output to be evaluated.
                keys should be:
                    input: input to the model
                    context: any context sent in with the input
                    output: output of the model that is being evaluated
                    metadata: dict of any fields that will pass through

        Returns:
            dict: _description_
        """
        logging.info(f"Evaluating single output: {output}")

    def evaluate_batch_output(self, outputs: list[dict]) -> list:
        """_summary_

        Args:
            outputs (dict]): _description_

        Returns:
            list: _description_
        """
        pass
