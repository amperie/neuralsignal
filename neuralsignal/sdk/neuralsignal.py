import logging
import json
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
from neuralsignal.core.modules.model_instrumentation\
    import load_model, generate_from_batch

logging.basicConfig(level=logging.INFO)


class SDK:
    """Main entrypoint into NeuralSignal SDK
    Configuration:
        - evaluation_mode: qb or direct
        - evaluators available: hallucination, bias, etc
        - S1 models for each
        - Prompts for each
        - Thresholds for each
        - Backend configuration
    Interfaces:
        - evaluate_output - qb evaluation of input/output
            - parameters: input/output/context/metadata
            - parameters: what detection to run (hallu/bias/etc)
        - evaluate_direct: instrumentation and real-time evaluation
            - wrap the generate function
    """

    default_config = {
        "evaluation_mode": "qb",  # qb or direct
        "qb_config": {
            "qb_model": "google/flan-t5-large",
            "zone_size:": 1024,
            "qb_batch_size": 1,
            "quantization": "int8",  # int4, int8, no_quantization
            "device": "cuda",
        },
        "save_scans": False,  # Save scans to backend
        "backend_config": {},  # Backend endpoint
        "detectors": [],  # detectors can't be empty
    }

    def __init_qb(self):
        logging.info("Initializing NeuralSignal in QB mode")
        model_cfg = {
            "model_name": self.cfg["qb_config"]["qb_model"],
            "device": self.cfg["qb_config"]["device"],
            "quantization": self.cfg["qb_config"]["quantization"],
        }
        logging.info(
            f"Loading model: {self.cfg['qb_config']['qb_model']}"
            f" with config: {model_cfg}")

        self.tokenizer, self.model = load_model(model_cfg)

    def __init_direct(self):
        pass

    def __init__(self, config: dict) -> None:
        """Initizalizes the NeuralSignal SDK

        Args:
            config (dict): Dictionary of configuration options.
            Possible options:
            [TODO: Add options here]
        """
        json_str = json.dumps(config, indent=4, sort_keys=False)
        log_string = highlight(json_str, JsonLexer(), TerminalFormatter())
        logging.info(
            f"Initializing NeuralSignal SDK with config: {log_string}")

        self.cfg = config
        self.mode = config["evaluation_mode"] == "qb"
        if self.mode == "qb":
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
            dict: Returns the same dictionary as the input with additional fields:
                - behavior: name of the behavior detected
                - score: score of the behavior detected
                - judgement: 0 or 1 depending on the threshold and score
                - correlation_id: unique id for the evaluation
        """
        logging.info(f"Evaluating single output: {output}")

    def evaluate_batch_output(self, outputs: list[dict]) -> list:
        """Evaluates a batch of outputs

        Args:
            output (dict): dictionary that contains the output to be evaluated.
                keys should be:
                    input: input to the model
                    context: any context sent in with the input
                    output: output of the model that is being evaluated
                    metadata: dict of any fields that will pass through

        Returns:
            dict: Returns the same dictionary as the input with additional fields:
                - behavior: name of the behavior detected
                - score: score of the behavior detected
                - judgement: 0 or 1 depending on the threshold and score
                - correlation_id: unique id for the evaluation
        """
        if self.mode != "qb":
            raise ValueError(
                "Evaluate_batch_output is only available in QB mode")
        logging.debug(f"Evaluating batch output: {outputs}")
        # Generate activity in qb
        gi = generate_from_batch(
            outputs, self.model, self.tokenizer)

        # Run the detectors on the activity

    def generate():
        pass
