import logging
import json
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
from neuralsignal.core.modules.model_instrumentation\
    import load_model, generate_from_batch
from neuralsignal.core.modules.detector import Detector
from neuralsignal.core.modules.generation_instance import GenerationInstance
from neuralsignal.core.modules.prompting import wrap_with_prompt
from neuralsignal.core.modules.utils import generate_uuid
from neuralsignal.backend.ns_backend import NSBackend
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
        self.correlation_id = None
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
        - max_new_tokens for generation
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
        raise NotImplementedError("Direct mode not implemented")

    def __init__(
            self, application_name, sub_application_name,
            config: dict = None,
            default_config_path: str = None) -> None:
        """Initizalizes the NeuralSignal SDK

        Args:
            config (dict): Dictionary of configuration options.
            This dictionary should be structured the same way the main
            yaml config file is structured. Any values in this dict
            will override the default
            values from the yaml file.
            Possible options:
            [TODO: Add options here]
            default_config_path: alternative path to config file
        """
        self.application_name = application_name
        self.sub_application_name = sub_application_name

        if default_config_path is None:
            default_config = yaml.safe_load(
                open("neuralsignal/sdk/neuralsignal_sdk.yaml"))
        else:
            default_config = yaml.safe_load(
                open(default_config_path))

        if config is None:
            config = default_config
        else:
            config = {**default_config, **config}

        try:
            json_str = json.dumps(config, indent=4, sort_keys=False)
            log_string = highlight(json_str, JsonLexer(), TerminalFormatter())
        except TypeError:
            log_string = str(config)

        logging.info(
            f"Initializing NeuralSignal SDK with config: {log_string}")

        self.cfg = config

        # If we're saving scans, initialize backend
        self.save_scans = config["save_scans"]
        if self.save_scans:
            config["backend_config"]["application_name"] =\
                self.application_name
            config["backend_config"]["sub_application_name"] =\
                self.sub_application_name
            self.backend = NSBackend(config)

        self.mode = config["evaluation_mode"]
        if self.mode == "indirect":
            self.__init_indirect()
        self.default_indirect_instrumentation_cfg =\
            config["indirect_instrumentation_config"]
        self.config = config

    def set_config(self, key: str, value: str):
        """Sets a configuration value

        Args:
            key (str): key to set
            value (str): value to set
        """
        self.cfg[key] = value

    def evaluate_batch_output(
            self, outputs: list[dict], detectors: list[Detector]
            ) -> list[GenerationInstance]:
        """Evaluates a batch of outputs

        Args:
            output (dict): dictionary that contains the output to be evaluated.
                keys should be:
                    input: input to the model (user's query)
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

        # TODO: for indirect mode, need to replace the prompt
        # with each detector's prompt so we need to explode the outputs
        # and run each detector on each output in a batch
        # ie: 2 outputs and 3 detectors=6 total evaluations and batch size=6
        prompted_outputs = []
        for output in outputs:
            for d in detectors:
                prompt = wrap_with_prompt(d.prompt, output)
                prompted_outputs.append(prompt)

        # Generate activity in indirect
        gis = generate_from_batch(
            prompted_outputs, self.model, self.tokenizer,
            instrumentation_cfg=self.default_indirect_instrumentation_cfg,
            max_new_tokens=self.config['max_new_tokens'],
            truncation_length=self.config['truncation_length'],
            )

        # Unpack the outputs in the same order and run the detectors on each
        # We need to make two data structures:
        # RetVal - return value with the results of the evaluation for each
        # but it doesn't contain internal data like the tensors
        # gis - contains all the data, including internal. This is
        # used for saving to the backend if required

        # Make them both simultaneously
        # Build retVal from scratch with only the data that is going
        # back to the user/caller
        # gis styas the same, just need to add the relevant detections

        batch_idx = 0
        retVal = []

        for output in outputs:
            rgi = GenerationInstance()
            rgi.data['input'] = output['input']
            rgi.data['output'] = output['output']
            if 'decoded_output' in output:
                rgi.data['decoded_output'] = output['decoded_output']
            if 'ground_truth' in output:
                rgi.data['ground_truth'] = output['ground_truth']
            else:
                rgi.data['ground_truth'] = None
            if 'metadata' in output:
                rgi.data['metadata'] = output['metadata']
            else:
                rgi.data['metadata'] = None
            if 'context' in output:
                rgi.data['context'] = output['context']
            else:
                rgi.data['context'] = None
            rgi.data['generation_correlation_id'] = generate_uuid()
            for d in detectors:
                curr = gis[batch_idx]
                if d.enabled:
                    # If the detector is enabled, run it on the output tensor
                    detection = d.detect(curr.data['outputs'])
                    detection.prompted_input = curr.data['input']
                    rgi.add_detection(detection)
                    gis[batch_idx].add_detection(detection)
                # Update all the individual gis entries
                gis[batch_idx].data['input'] = output['input']
                gis[batch_idx].data['output'] = output['output']
                if 'ground_truth' in output:
                    gis[batch_idx].data['ground_truth'] =\
                        output['ground_truth']
                else:
                    gis[batch_idx].data['ground_truth'] = None
                if 'metadata' in output:
                    gis[batch_idx].data['metadata'] = output['metadata']
                else:
                    gis[batch_idx].data['metadata'] = None
                if 'context' in output:
                    gis[batch_idx].data['context'] = output['context']
                else:
                    gis[batch_idx].data['context'] = None
                gis[batch_idx].data['generation_correlation_id'] =\
                    rgi.data['generation_correlation_id']
                gis[batch_idx].add_data_to_save(
                    {"evaluation_method": "indirect"})
                batch_idx += 1
            retVal.append(rgi)

        # At this point gis contains the consistent data for each detector run
        # gis should be used for the purposes of saving to the backend
        # retVal's readings are only for the first detector

        # Save to the backend if required
        if self.save_scans:
            for gi in gis:
                self.backend.save_scan(gi)

        return retVal

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
            dr.correlation_id = gi.data['generation_correlation_id']
            dr.detections = gi.detections
            retVal.append(dr)
        return retVal

    def generate():
        pass
