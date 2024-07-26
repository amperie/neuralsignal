import logging
import json
import os
from torch.cuda import OutOfMemoryError
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
from neuralsignal.core.modules.neuralsignal_config import sdk_config

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
        # TODO: pass through the config directly instead of picking them out
        model_cfg = {
            "model_name": self.cfg["indirect_config"]["indirect_model"],
            "zone_size": self.cfg["indirect_config"]["zone_size"],
            "device": self.cfg["indirect_config"]["device"],
            "quantization": self.cfg["indirect_config"]["quantization"],
        }

        self.model_config = model_cfg
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
        self.enabled = True
        self.disabled_reason = ""

        if default_config_path is None:
            dirname = os.path.dirname(__file__)
            default_config_path = os.path.join(
                dirname, './neuralsignal_sdk.yaml')
            default_config = yaml.safe_load(
                open(default_config_path))
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
        self.dynamic_batch_size = None
        self.use_dynamic_batch_size = config["use_dynamic_batch_size"]
        self.oom_count = 0
        self.max_oom_count = config["max_oom_count"]

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

    def _evaluate_batch_output(
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

        # Check if SDK is disabled and return empty list if so
        if not self.check_enabled():
            return []

        if self.mode != "indirect":
            raise ValueError(
                "Evaluate_batch_output is only available in indirect mode")
        logging.debug(f"Evaluating batch output: {outputs}")

        # TODO: BUG HERE
        # The tokenizer pads the batches to the size of the biggest prompt
        # The S1 model gives different scores depending on the batch size
        # Need to figure out if they can still be batched without padding
        # or is they'll need to be passed in series
        prompted_outputs = []
        for output in outputs:
            for d in detectors:
                prompt = wrap_with_prompt(d.prompt, output)
                prompted_outputs.append(prompt)

        # torch.cuda.OutOfMemoryError is possible here
        # Generate activity in indirect
        # Catch errors for logging then re-raise them
        try:
            gis = generate_from_batch(
                prompted_outputs, self.model, self.tokenizer,
                instrumentation_cfg=self.default_indirect_instrumentation_cfg,
                max_new_tokens=self.config['max_new_tokens'],
                truncation_length=self.config['truncation_length'],
                )
        except OutOfMemoryError:
            raise OutOfMemoryError(
                "NeuralSignal out of memory error. "
            )
        except TypeError as e:
            logging.error("Suppressing TypeError in generate_from_batch")
            logging.error(
                f"Parameters:\n prompted_outputs: {prompted_outputs}\n\n"
                f"inst_cfg: {self.default_indirect_instrumentation_cfg}\n\n"
                )
            logging.error(f"{e}")
        except RuntimeError as e:
            logging.error("Suppressing RuntimeError in generate_from_batch")
            logging.error(
                f"Parameters:\n prompted_outputs: {prompted_outputs}\n\n"
                )
            logging.error(f"{e}")
            # TODO: Add more specific error handling
            return []

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
                    detection = d.detect(
                        curr.data['outputs'],
                        current_zone_size=curr.data['zone_size'],
                        target_zone_size=self.config['zone_size']
                        )
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
        # retVal's output readings are only for the first detector

        # Save to the backend if required
        if self.save_scans:
            for gi in gis:
                self.backend.save_scan(gi)

        return retVal

    def evaluate_indirect_output_old(
            self, outputs: list[dict], detectors: list[Detector]
            ) -> list[DetectionResults]:

        # Check if SDK is disabled and return empty list if so
        if not self.check_enabled():
            return []

        gis = self._evaluate_batch_output(outputs, detectors)
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

    # TODO: Need to make this method without requiring
    # detector list. Get the detector list from
    # the config file instead or only ask for a list
    # of strings of the detector names

    def evaluate_indirect(
            self, outputs: list[dict], detectors: list[str]
            ) -> list[DetectionResults]:
        # TODO: cache the detector object so they don't load each time
        # this method runs
        dl = []
        for d in detectors:
            cfg = sdk_config.get_detector_config(d)
            cfg['application_name'] = self.application_name
            cfg['sub_application_name'] = self.sub_application_name
            dl.append(Detector(cfg))

        return self.evaluate_indirect_output(outputs, dl)

    def evaluate_indirect_output(
            self, outputs: list[dict], detectors: list[Detector]
            ) -> list[DetectionResults]:

        # Check if SDK is disabled and return empty list if so
        if not self.check_enabled():
            return []
        # Dynamic batch size. Run the whole thing first
        # If OOM, half the batch size until no OOM happens
        oom = True
        # If we're not using dynamic batch size, batch size will
        # always start with the original value
        if not self.use_dynamic_batch_size:
            batch_size = len(outputs)
            original_batch_size = batch_size
        else:
            # If we're using dynamic batch size, check if we've
            # established a dynamic batch size before and use that
            if self.dynamic_batch_size is None:
                batch_size = len(outputs)
                original_batch_size = batch_size
            else:
                # If we already have a dynamic batch size
                # that we found in a previous iteration
                # and dynamic batch size is enabled, use that
                original_batch_size = self.dynamic_batch_size
                batch_size = self.dynamic_batch_size
        start_idx = 0
        gis = []
        testing = False
        while oom:
            try:
                # TODO: this will cause data to repeat itself
                # If a smaller batch other than the first one fails
                # because the first one will already be saved and this will
                # rerun everything again with a smaller batch size
                while start_idx < original_batch_size:
                    if testing:
                        raise OutOfMemoryError
                    end_idx = start_idx + batch_size
                    res =\
                        self._evaluate_batch_output(
                            outputs[
                                start_idx:min(end_idx, original_batch_size)],
                            detectors)
                    gis = gis + res
                    # TODO: start_idx and end_idx can be used to fix
                    # the problem of data repeating
                    start_idx += batch_size
                    end_idx = start_idx + batch_size
                # If we get here, we made it through the batch
                # so we'l save the batch size for later use
                # and set oom to false to exit the loop
                self.dynamic_batch_size = batch_size
                oom = False
            except OutOfMemoryError as e:
                # If we're already at batch size = 1 there's nowhere else to go
                if batch_size == 1:
                    logging.error("CUDA OOM on batch size of 1")
                    # raise OutOfMemoryError(
                    #    f"CUDA OOM on batch size of 1, FATAL {e}")

                logging.error(f"CUDA OOM on batch size of {batch_size}")
                batch_size = max(1, int(batch_size/2))
                logging.error(f"Dropping batch size to {batch_size}")
                logging.error(f"CUDA Error: {e}")
                self.oom_count += 1
                if self.oom_count > self.max_oom_count:
                    logging.fatal("Exceeded MAX_OOM_COUNT, SDK disabled")
                    self.disable_sdk("Exceeded MAX_OOM_COUNT") 
                    raise OutOfMemoryError(
                        f"Exceeded MAX_OOM_COUNT, SDK disabled. FATAL {e}")
                else:
                    logging.info(
                        f"Reloading model and setting batch size to "
                        f"{batch_size}")
                    self.tokenizer, self.model = load_model(self.model_config)

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
        raise NotImplementedError

    def disable_sdk(self, reason: str):
        self.enabled = False
        self.disabled_reason = reason

    def check_enabled(self):
        if not self.enabled:
            logging.error("SDK is not enabled")
        return self.enabled
