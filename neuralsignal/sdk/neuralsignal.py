"""
neuralsignal.sdk.neuralsignal
==============================
Primary public API for the NeuralSignal SDK.

Typical usage::

    from neuralsignal.sdk.neuralsignal import SDK

    sdk = SDK(
        application_name="my_app",
        sub_application_name="chatbot",
    )

    results = sdk.evaluate_indirect(
        outputs=[
            {
                "input": "What is the capital of France?",
                "output": "Paris",
                "context": "France is a country in Western Europe.",
            }
        ],
        detectors=["hallucination"],
    )

    for r in results:
        for behavior, det in r.detections.items():
            print(behavior, det.score)

Classes
-------
- :class:`DetectionResults` Ã¢â‚¬â€ data-only container returned to callers.
- :class:`SDK` Ã¢â‚¬â€ main entry point; initialises the judge model and exposes
  evaluation methods.
"""

import logging
import json
from dataclasses import dataclass, field
from torch.cuda import OutOfMemoryError
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
from neuralsignal.core.modules.model_instrumentation\
    import load_model, generate_from_batch
from neuralsignal.core.modules.tensors import subtract_scans
from neuralsignal.core.modules.detector import Detector
from neuralsignal.core.modules.generation_instance import GenerationInstance
from neuralsignal.core.modules.prompting import wrap_with_prompt
from neuralsignal.core.modules.utils import generate_uuid
from neuralsignal.backend.ns_backend import NSBackend
from neuralsignal.config.loader import load_sdk_config

logging.basicConfig(level=logging.INFO)


class EvaluationError(RuntimeError):
    """Raised when evaluation fails for reasons other than OOM."""


@dataclass
class DetectionResults:
    """Public, data-only container that carries evaluation results back to the
    caller.

    Each instance corresponds to one input/output pair that was evaluated.
    Detection scores for individual behaviors are stored in :attr:`detections`,
    keyed by behavior name (e.g. ``"hallucination"``).

    Attributes
    ----------
    input : str or None
        The original user query / prompt that was evaluated.
    output : str or None
        The LLM-generated text that was evaluated.
    ground_truth : str or None
        Optional ground-truth answer, when available from the caller.
    metadata : dict or None
        Arbitrary caller-supplied metadata that passes through unchanged.
    correlation_id : str or None
        UUID that ties this result to the corresponding scan stored in the
        backend.
    detections : dict
        Mapping of ``behavior_name Ã¢â€ â€™ DetectionResults`` (the internal detector
        variant from :mod:`neuralsignal.core.modules.detector`).  Each value
        exposes ``.score``, ``.threshold``, and ``.correlation_id``.
    """

    input: str | None = None
    output: str | None = None
    ground_truth: str | None = None
    metadata: dict | None = None
    correlation_id: str | None = None
    detections: dict = field(default_factory=dict)

    def __str__(self):
        retVal = f"DetectionResults: {self.input} - {self.output}"\
                f"- {self.ground_truth} - {self.metadata}"
        return retVal


class SDK:
    """Main entry point for the NeuralSignal SDK.

    Initialises a judge model (indirect mode) and exposes methods for
    evaluating LLM input/output pairs against one or more behavior detectors.

    Evaluation modes
    ----------------
    indirect
        A separate "judge" LLM (default: ``google/flan-t5-large``) receives a
        templated prompt that embeds the original input/output and context.
        Internal activations are captured by PyTorch forward hooks, featurized
        into "scans", and scored by a pre-trained S1 classifier.
    direct
        Real-time instrumentation of the application's own LLM.
        **Not yet implemented.**

    OOM handling
    ------------
    When :attr:`use_dynamic_batch_size` is ``True`` (default) the SDK
    automatically halves the batch size after each ``OutOfMemoryError`` and
    reloads the model.  If the OOM count exceeds :attr:`max_oom_count` the SDK
    disables itself permanently for the lifetime of the object and raises a
    fatal ``OutOfMemoryError``.

    Parameters
    ----------
    application_name : str
        Logical name of the calling application.  Stored with every scan in
        the backend.
    sub_application_name : str
        Sub-section or feature within the application.  Stored with every scan.
    config : dict, optional
        Override dictionary merged on top of the YAML defaults.  Keys mirror
        the structure of ``neuralsignal_sdk.yaml``.
    default_config_path : str, optional
        Path to an alternative YAML config file.  When omitted the bundled
        ``neuralsignal_sdk.yaml`` next to this module is used.

    Attributes
    ----------
    enabled : bool
        ``False`` after :meth:`disable_sdk` is called (e.g. on fatal OOM).
    disabled_reason : str
        Human-readable explanation set when the SDK is disabled.
    mode : str
        Active evaluation mode (``"indirect"`` or ``"direct"``).
    save_scans : bool
        Whether scans are persisted to the configured backend.
    dynamic_batch_size : int or None
        Stable batch size discovered by the OOM-halving loop; ``None`` until
        the first successful batch.
    oom_count : int
        Cumulative count of ``OutOfMemoryError`` occurrences since init.
    """

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def __init_indirect(self):
        """Load the judge model and tokenizer for indirect evaluation.

        Reads ``indirect_config`` from the merged config dict and calls
        :func:`~neuralsignal.core.modules.model_instrumentation.load_model`.
        Sets ``self.tokenizer`` and ``self.model``.
        """
        logging.info("Initializing NeuralSignal in indirect mode")
        # TODO: pass through the config directly instead of picking them out
        model_cfg = {
            "model_name": self.cfg["indirect_config"]["indirect_model"],
            "device": self.cfg["indirect_config"]["device"],
            "quantization": self.cfg["indirect_config"]["quantization"],
        }

        self.model_config = model_cfg
        self.tokenizer, self.model = load_model(model_cfg)

    def __init_direct(self):
        """Placeholder for direct-mode initialisation.

        Raises
        ------
        NotImplementedError
            Always; direct mode is not yet implemented.
        """
        raise NotImplementedError("Direct mode not implemented")

    def __init__(
            self, application_name, sub_application_name,
            config: dict = None,
            default_config_path: str = None) -> None:
        """Initialise the NeuralSignal SDK.

        Loads configuration, optionally merging caller-supplied overrides on
        top of the YAML defaults.  Initialises the judge model (indirect mode)
        and, when ``save_scans`` is ``True``, connects to the configured
        storage backend.

        Parameters
        ----------
        application_name : str
            Logical name of the calling application.
        sub_application_name : str
            Sub-section or feature within the application.
        config : dict, optional
            Key/value overrides merged onto the YAML defaults.  See
            ``neuralsignal_sdk.yaml`` for the full list of supported keys.
        default_config_path : str, optional
            Absolute or relative path to an alternative YAML config file.
        """
        self.application_name = application_name
        self.sub_application_name = sub_application_name
        self.enabled = True
        self.disabled_reason = ""

        resolved_config = load_sdk_config(
            config_path=default_config_path,
            overrides=config,
        )
        config = resolved_config.data

        try:
            json_str = json.dumps(config, indent=4, sort_keys=False)
            log_string = highlight(json_str, JsonLexer(), TerminalFormatter())
        except TypeError:
            log_string = str(config)

        logging.info(
            f"Initializing NeuralSignal SDK with config: {log_string}")

        self.resolved_config = resolved_config
        self.cfg = config
        self.dynamic_batch_size = None
        self.use_dynamic_batch_size = config["use_dynamic_batch_size"]
        self.oom_count = 0
        self.max_oom_count = config["max_oom_count"]

        # If we're saving scans, initialize backend
        self.save_scans = config["save_scans"]
        if self.save_scans:
            self.backend = NSBackend({
                "application_name": self.application_name,
                "sub_application_name": self.sub_application_name,
                "backend_config": self.resolved_config.get_backend_config(),
            })

        self.mode = config["evaluation_mode"]
        if self.mode == "indirect":
            self.__init_indirect()
        self.default_indirect_instrumentation_cfg =\
            config["indirect_instrumentation_config"]
        self.config = config

    def _get_detector_config(self, detector_name: str) -> dict:
        return self.resolved_config.get_detector_config(detector_name)

    @staticmethod
    def _build_public_results(
            gis: list[GenerationInstance]) -> list[DetectionResults]:
        results = []
        for gi in gis:
            results.append(DetectionResults(
                input=gi.data.get("input"),
                output=gi.data.get("output"),
                ground_truth=gi.data.get("ground_truth"),
                metadata=gi.data.get("metadata"),
                correlation_id=gi.data.get("generation_correlation_id"),
                detections=dict(gi.detections),
            ))
        return results

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_config(self, key: str, value: str):
        """Set a single top-level configuration value at runtime.

        Parameters
        ----------
        key : str
            Top-level key in the active config dict.
        value : str
            New value to assign.

        Notes
        -----
        Changes are applied to the in-memory config only and are not persisted
        to the YAML file.
        """
        self.cfg[key] = value

    # ------------------------------------------------------------------
    # Internal evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_batch_output(
            self, outputs: list[dict], detectors: list[Detector]
            ) -> list[GenerationInstance]:
        """Run the judge model over a batch and apply detectors to each scan.

        For each (output, detector) pair a prompt is built with
        :func:`~neuralsignal.core.modules.prompting.wrap_with_prompt` and fed
        through the judge model.  Activation tensors are captured by the
        instrumentation hooks and stored in :class:`GenerationInstance` objects.
        Each enabled detector then scores its corresponding scan.

        Parameters
        ----------
        outputs : list[dict]
            Each dict must contain:

            * ``"input"`` (str) Ã¢â‚¬â€ user query.
            * ``"output"`` (str) Ã¢â‚¬â€ LLM response to evaluate.

            Optional keys:

            * ``"context"`` (str) Ã¢â‚¬â€ background text for grounded tasks.
            * ``"ground_truth"`` (str) Ã¢â‚¬â€ expected answer.
            * ``"metadata"`` (dict) Ã¢â‚¬â€ arbitrary pass-through data.
            * ``"decoded_output"`` (str) Ã¢â‚¬â€ token-decoded form of the output.

        detectors : list[Detector]
            Detector objects (each wraps an S1 model + prompt template).
            ``scan_delta`` detectors consume two consecutive batch slots.

        Returns
        -------
        list[GenerationInstance]
            One instance per (output Ãƒâ€” detector) pair, each populated with
            scan data and detection scores.  Returns an empty list when the
            SDK is disabled.

        Raises
        ------
        OutOfMemoryError
            Re-raised after logging when the judge model runs out of GPU
            memory.

        Notes
        -----
        **Known bug**: the tokenizer pads the batch to the longest prompt,
        which causes the S1 model to return different scores depending on
        batch size.  Consider passing inputs in series until this is resolved.
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
                # Check to see if it's a scan delta type
                # If so, split the prompt and put it into a batch of 2
                if d.type == "scan_delta":
                    prompts = prompt.split("<<--separator-->>")
                    prompt1 = prompts[0]
                    prompt2 = prompts[1]
                    prompted_outputs.append(prompt1)
                    prompted_outputs.append(prompt2)
                else:
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
        except (TypeError, RuntimeError) as e:
            logging.error(
                "Evaluation failed in generate_from_batch with prompts: %s",
                prompted_outputs)
            raise EvaluationError(
                "NeuralSignal evaluation failed during generation"
            ) from e

        # Unpack the outputs in the same order and run the detectors on each
        # We need to make two data structures:
        # RetVal - return value with the results of the evaluation for each
        # but it doesn't contain internal data like the tensors
        # gis - contains all the data, including internal. This is
        # used for saving to the backend if required

        # Make them both simultaneously
        # Build retVal from scratch with only the data that is going
        # back to the user/caller
        # gis stays the same, just need to add the relevant detections

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
                    detection = d.detect(curr.data)
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
            if d.type == "scan_delta":
                self.backend.save_scan(
                    subtract_scans(gis[0], gis[1])
                )
            else:
                for gi in gis:
                    self.backend.save_scan(gi)

        return retVal

    def evaluate_indirect_output_old(
            self, outputs: list[dict], detectors: list[Detector]
            ) -> list[DetectionResults]:
        """Deprecated wrapper around :meth:`_evaluate_batch_output`.

        .. deprecated::
            Use :meth:`evaluate_indirect_output` instead.

        Parameters
        ----------
        outputs : list[dict]
            See :meth:`_evaluate_batch_output`.
        detectors : list[Detector]
            Pre-constructed detector objects.

        Returns
        -------
        list[DetectionResults]
            One result per output item.
        """

        # Check if SDK is disabled and return empty list if so
        if not self.check_enabled():
            return []

        gis = self._evaluate_batch_output(outputs, detectors)
        return self._build_public_results(gis)

    # TODO: Need to make this method without requiring
    # detector list. Get the detector list from
    # the config file instead or only ask for a list
    # of strings of the detector names

    def evaluate_indirect(
            self, outputs: list[dict], detectors: list[str]
            ) -> list[DetectionResults]:
        """Evaluate a list of outputs by detector name (preferred API).

        Looks up each named detector in the active config via
        :meth:`~neuralsignal.core.modules.neuralsignal_config.NeuralSignalConfig.get_detector_config`,
        constructs :class:`~neuralsignal.core.modules.detector.Detector`
        objects, and delegates to :meth:`evaluate_indirect_output`.

        Parameters
        ----------
        outputs : list[dict]
            Each dict must contain ``"input"`` and ``"output"`` keys.
            Optional: ``"context"``, ``"ground_truth"``, ``"metadata"``.
        detectors : list[str]
            Behavior names to run, e.g. ``["hallucination", "input_toxicity"]``.
            Each name must match a ``behavior_name`` entry in the config's
            ``detectors`` list.

        Returns
        -------
        list[DetectionResults]
            One :class:`DetectionResults` per output item.

        Notes
        -----
        TODO: detector objects are re-instantiated on every call.  Cache them
        to avoid repeated model loading overhead.
        """
        # TODO: cache the detector object so they don't load each time
        # this method runs
        dl = []
        for d in detectors:
            cfg = self._get_detector_config(d)
            cfg['application_name'] = self.application_name
            cfg['sub_application_name'] = self.sub_application_name
            dl.append(Detector(cfg))

        return self.evaluate_indirect_output(outputs, dl)

    def evaluate_indirect_output(
            self, outputs: list[dict], detectors: list[Detector]
            ) -> list[DetectionResults]:
        """Evaluate a list of outputs using pre-constructed detector objects.

        Implements automatic OOM recovery via dynamic batch-size halving.
        The entire batch is processed first; on ``OutOfMemoryError`` the
        batch size is halved and the model is reloaded.  The discovered stable
        batch size is persisted in :attr:`dynamic_batch_size` for subsequent
        calls.

        Parameters
        ----------
        outputs : list[dict]
            Each dict must contain ``"input"`` and ``"output"`` keys.
            Optional: ``"context"``, ``"ground_truth"``, ``"metadata"``.
        detectors : list[Detector]
            Pre-constructed :class:`~neuralsignal.core.modules.detector.Detector`
            objects.  Use :meth:`evaluate_indirect` to build these from names.

        Returns
        -------
        list[DetectionResults]
            One :class:`DetectionResults` per output item.  Returns ``[]``
            when the SDK is disabled.

        Raises
        ------
        OutOfMemoryError
            When OOM persists at batch size 1 and :attr:`max_oom_count` is
            exceeded; the SDK is permanently disabled before raising.

        Notes
        -----
        **Known bug**: if an OOM occurs after some sub-batches have already
        been saved to the backend, the retry loop will re-process and re-save
        those items when it restarts from ``start_idx = 0``.
        """

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

        return self._build_public_results(gis)

    # ------------------------------------------------------------------
    # Stubs / future API
    # ------------------------------------------------------------------

    def generate():
        """Stub for future direct-mode generation wrapping.

        Raises
        ------
        NotImplementedError
            Always; not yet implemented.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # SDK lifecycle
    # ------------------------------------------------------------------

    def disable_sdk(self, reason: str):
        """Permanently disable the SDK for this instance.

        Called automatically when :attr:`max_oom_count` is exceeded.  May
        also be called manually to suppress all evaluations without raising
        an error.

        Parameters
        ----------
        reason : str
            Human-readable explanation stored in :attr:`disabled_reason`.
        """
        self.enabled = False
        self.disabled_reason = reason

    def check_enabled(self):
        """Return whether the SDK is currently enabled.

        Logs an error when disabled.

        Returns
        -------
        bool
            ``True`` if evaluation should proceed; ``False`` if the SDK has
            been disabled (e.g. after repeated OOM errors).
        """
        if not self.enabled:
            logging.error("SDK is not enabled")
        return self.enabled


