import logging
import json
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
from neuralsignal.core.modules.model_instrumentation\
    import load_model, generate_from_batch
from neuralsignal.core.modules.detector import Detector
from neuralsignal.core.modules.generation_instance import GenerationInstance
import yaml

logging.basicConfig(level=logging.INFO)


class DetectionResults:
    """Main container to provide detection results back
    to the user of the SDK. Only contains data, no methods
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.ground_truth = None
        self.metadata = None
        self.detections = []

    def __str__(self):
        retVal = f"DetectionResults: {self.input} - {self.output}"\
                f"- {self.ground_truth} - {self.metadata} - {self.detections}"
        return retVal


class SDK:
    """Main entrypoint into NeuralSignal SDK
    Configuration:
        - evaluation_mode: indirect or direct
        - evaluators available: hallucination, bias, etc
        - S1 models for each
        - Prompts for each
        - Thresholds for each
        - Backend configuration
    Interfaces:
        - evaluate_output - indirect evaluation of input/output
            - parameters: input/output/context/metadata
            - parameters: what detection to run (hallu/bias/etc)
        - evaluate_direct: instrumentation and real-time evaluation
            - wrap the generate function
    """

    def __init_indirect(self):
        logging.info("Initializing NeuralSignal in indirect mode")
        model_cfg = {
            "model_name": self.cfg["indirect_config"]["indirect_model"],
            "device": self.cfg["indirect_config"]["device"],
            "quantization": self.cfg["indirect_config"]["quantization"],
        }

        self.tokenizer, self.model = load_model(model_cfg)

    def __init_direct(self):
        pass

    def __init__(self, config: dict = None) -> None:
        """Initizalizes the NeuralSignal SDK

        Args:
            config (dict): Dictionary of configuration options.
            Possible options:
            [TODO: Add options here]
        """

        default_config = yaml.safe_load(
            open("neuralsignal/sdk/neuralsignal_sdk.yaml"))
        if config is None:
            config = default_config
        else:
            config = {**default_config, **config}

        json_str = json.dumps(config, indent=4, sort_keys=False)
        log_string = highlight(json_str, JsonLexer(), TerminalFormatter())
        logging.info(
            f"Initializing NeuralSignal SDK with config: {log_string}")

        self.cfg = config
        self.mode = config["evaluation_mode"]
        if self.mode == "indirect":
            self.__init_indirect()
        self.default_indirect_instrumentation_cfg =\
            config["indirect_instrumentation_config"]

    def evaluate_batch_output(
            self, outputs: list[dict], detectors: list[Detector]
            ) -> list[GenerationInstance]:
        """Evaluates a batch of outputs

        Args:
            output (dict): dictionary that contains the output to be evaluated.
                keys should be:
                    input: input to the model
                    context: any context sent in with the input
                    output: output of the model that is being evaluated
                    metadata: dict of any fields that will pass through

        Returns:
            dict: Returns the same dictionary as the input with
            additional fields:
                - behavior: name of the behavior detected
                - score: score of the behavior detected
                - judgement: 0 or 1 depending on the threshold and score
                - correlation_id: unique id for the evaluation
        """
        if self.mode != "indirect":
            raise ValueError(
                "Evaluate_batch_output is only available in indirect mode")
        logging.debug(f"Evaluating batch output: {outputs}")
        # Generate activity in indirect
        gis = generate_from_batch(
            outputs, self.model, self.tokenizer,
            instrumentation_cfg=self.default_indirect_instrumentation_cfg
            )

        # Run the detectors on the activity
        for batch_idx in range(len(gis)):
            for detector in detectors:
                if detector.enabled:
                    d = detector.detect(gis[batch_idx].data['outputs'])
                    gis[batch_idx].add_detection(d)

        # Setup the results to include the list of outputs
        # with the behavior, score, judgement, and correlation_id
        return gis

    def evaluate_indirect_output(
            self, outputs: list[dict], detectors: list[Detector]
            ) -> list[DetectionResults]:

        gis = self.evaluate_batch_output(outputs, detectors)
        retVal = []
        for gi in gis:
            dr = DetectionResults()
            dr.input = gi.data['input']
            dr.output = gi.data['output']
            dr.ground_truth = gi.data['ground_truth']
            dr.metadata = gi.data['metadata']
            dr.detections = gi.detections
            retVal.append(dr)
        return retVal

    def generate():
        pass
