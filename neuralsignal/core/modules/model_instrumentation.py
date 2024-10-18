import logging
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import BitsAndBytesConfig
from neuralsignal.core.modules.generation_instance import GenerationInstance
from neuralsignal.core.modules.collector import Collector
from transformers.models.t5.modeling_t5 import T5LayerFF
from transformers.models.t5.modeling_t5 import T5LayerSelfAttention
from transformers.models.t5.modeling_t5 import T5LayerCrossAttention
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoModelForMaskedLM

logging.basicConfig(level=logging.INFO)


def load_model(model_config: dict) -> tuple[AutoTokenizer, AutoModel]:
    """Loads a model from a config, has functions to instrument model
    and to run generation on model

    Args:
        model_config (dict): dictionary containing the model configuration
        model_config should contain:
            model_name: name of the model to load
            device: device to load the model on ("cuda:0")
            quantization: int4, int8, no_quantization

    Returns:
        AutoModel: loaded model from HuggingFace
    """
    logging.info(
        f"Loading model: {model_config['model_name']} with "
        f"config: {model_config}")
    model_type = get_model_type(model_config["model_name"])

    if model_type == "mpnet":
        model = AutoModelForMaskedLM.from_pretrained(
            model_config["model_name"],
            device_map=model_config["device"],
            trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_name"])
        return tokenizer, model

    if model_config["quantization"] == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )

        if model_type == "t5":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config["model_name"],
                device_map=model_config["device"],
                quantization_config=bnb_config,
                trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config["model_name"],
                device_map=model_config["device"],
                quantization_config=bnb_config,
                trust_remote_code=True)

        logging.info(
            f"Loaded {model_config['model_name']} in 4 bit quantization")

    elif model_config["quantization"] == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16
            )

        if model_type == "t5":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config["model_name"],
                device_map=model_config["device"],
                quantization_config=bnb_config,
                trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config["model_name"],
                device_map=model_config["device"],
                quantization_config=bnb_config,
                trust_remote_code=True)

        logging.info(
            f"Loaded {model_config['model_name']} in 8 bit quantization")

    else:
        if model_type == "t5":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config["model_name"],
                device_map=model_config["device"])
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config["model_name"],
                device_map=model_config["device"])
        logging.info(
            f"Loaded {model_config['model_name']} unquantized")

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    logging.info(
        f"Loaded {model_config['model_name']} tokenizer")

    return (tokenizer, model)


def generate_from_batch(
        input: list[str], model: AutoModel, tokenizer: AutoTokenizer,
        instrumentation_cfg: dict = None, truncation_length: int = 0,
        max_new_tokens: int = 128,
        ) -> list[GenerationInstance]:
    """Generates a response from a model for a given string array

    Args:
        input (str): input to the model
        model (AutoModel): model to generate the response
        tokenizer (AutoTokenizer): tokenizer to tokenize the input
        (optional) instrumentation_cfg (dict): configuration for instrumenting
            To use default values, pass in an empty dict {}
            Should contain: instrument_encoder, instrument_decoder,
            instrument_FF, instrument_attention, instrument_embedding and
            collector_config dict. If ommited, default values are used:
                "instrument_encoder": True,
                "instrument_decoder": True,
                "instrument_FF": True,
                "instrument_attention": True,
                "instrument_embedding": True,
                "collector_config": {mode: "additive",
                data_to_save: ["output", "module", "layer_info",
                "topology"], zone_size: 512}

    Returns:
        [GenerationInstance]: Returns a list of GenerationInstance objects
        that contain all the details collected for each input
    """
    logging.debug(f"Generating response for input: {input}")
    # Setup for instrumenting model
    # If collector exists, we're instrumenting
    default_instrumentation_cfg = {
            "instrument_encoder": True,
            "instrument_decoder": True,
            "instrument_FF": True,
            "instrument_attention": True,
            "instrument_embedding": True,
            "collector_config": {
                "mode": "additive",
                "data_to_save": ["outputs", "layer_info", "topology"],
                "zone_size": 512,
            },
        }

    batch_size = len(input)
    model_instrumented = False
    if instrumentation_cfg is not None:
        instrumentation_cfg = {
            **default_instrumentation_cfg, **instrumentation_cfg}

        hc = Collector(instrumentation_cfg["collector_config"])
        hndls = instrument_model(instrumentation_cfg, model, hc)
        model_instrumented = True

    # Generation

    input_list = input

    if batch_size > 1:
        # If batch size>1 we need padding
        input_ids = tokenizer(
            input_list, return_tensors="pt",
            padding=True, truncation=True).input_ids
    elif truncation_length == 0:
        # IF batch = 1 then see if we're truncating
        input_ids = tokenizer(
            input_list, return_tensors="pt",
            padding=False, truncation=False).input_ids
    else:
        input_ids = tokenizer(
            input_list, return_tensors="pt",
            padding=False, truncation=True,
            max_length=truncation_length).input_ids

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    try:
        input_ids = input_ids.to(model.device)
        output = model.generate(
            input_ids, pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens
            )
    except Exception as e:
        if model_instrumented:
            deinstrument_model(hndls)
            model_instrumented = False
        raise e

    if model_instrumented:
        hc.finish_and_get_data()

    retVal = []
    for batch_idx in range(batch_size):
        decoded_output = tokenizer.decode(
            output[batch_idx], skip_special_tokens=True)
        gi = GenerationInstance({
            "input": input[batch_idx],
            "output": decoded_output,
            "model_name": model.name_or_path
        })
        gi.add_data({"decoded_output": decoded_output})
        if model_instrumented:
            data = hc.get_data_by_batch_index(batch_idx)
            # Add data to return value
            gi.add_data(data)
        retVal.append(gi)

    if model_instrumented:
        deinstrument_model(hndls)
        model_instrumented = False

    logging.debug(
        f"Generated response for batch size {batch_size} "
        f"input: {input} \n\n{decoded_output}")

    return retVal


def generate_from_string(
        input: str, model: AutoModel, tokenizer: AutoTokenizer,
        instrumentation_cfg: dict = None, truncate: bool = False,
        ) -> list[GenerationInstance]:
    """Generates a response from a model for a given string

    Args:
        input (str): input to the model
        model (AutoModel): model to generate the response
        tokenizer (AutoTokenizer): tokenizer to tokenize the input
        (optional) instrumentation_cfg (dict): configuration for instrumenting
            To use default values, pass in an empty dict {}
            Should contain: instrument_encoder, instrument_decoder,
            instrument_FF, instrument_attention, instrument_embedding and
            collector_config dict. If ommited, default values are used:
                "instrument_encoder": True,
                "instrument_decoder": True,
                "instrument_FF": True,
                "instrument_attention": True,
                "instrument_embedding": True,
                "collector_config": {mode: "additive",
                data_to_save: ["output", "module", "layer_info",
                "topology"], zone_size: 512}

    Returns:
        str: generated response
    """
    return generate_from_batch(
        [input], model, tokenizer,
        instrumentation_cfg, truncate)


def get_model_type(model) -> str:
    if isinstance(model, str):
        model_name = model
    else:
        model_name = model.name_or_path
    if "t5" in model_name:
        return "t5"
    if "bert" in model_name:
        return "bert"
    if "mpnet" in model_name:
        return "mpnet"
    if "flan" in model_name:
        return "t5"
    if "JudgeLM" in model_name:
        return "llama2"
    if "Llama" in model_name:
        return "llama2"
    if model_name.startswith("meta-llama"):
        return "llama2"
    if "Mixtral-8x" in model_name:
        return "mixtral8x"
    if "Mistral-7B" in model_name:
        return "mistral7b"
    if "Phi-3" in model_name:
        return "phi3"
    raise ValueError(f"Model type not implemented: {model_name}")


def instrument_model(cfg, model, hc: Collector) -> list:
    """Instruments a local model
        Instrumentation config dictionary should include:
        model_type: t5, llama2, mixtral8x, mistral7b
        instrument_encoder: bool
        instrument_decoder: bool
        instrument_FF: bool
        instrument_attention: bool
        instrument_embedding: bool
    Args:
        cfg (dict): Instrumentation configuration
        model (_type_): the model object
        hc (Collector): collector object

    Returns:
        list: _description_
    """
    type = get_model_type(model)
    logging.debug(f"Instrumenting model of type: {type}")

    if type == "t5":
        return instrument_t5(cfg, model, hc)
    if type == "mpnet":
        return instrument_mpnet(cfg, model, hc)
    if type == "bert":
        return instrument_bert(cfg, model, hc)
    if type == "llama2":
        return instrument_llama2(cfg, model, hc)
    if type == "mixtral8x":
        return instrument_mixtral_8x(cfg, model, hc)
    if type == "mistral7b":
        return instrument_mistral_7b(cfg, model, hc)
    if type == "phi3":
        return instrument_phi3(cfg, model, hc)
    return None


def deinstrument_model(registered_hooks: list) -> list:
    for hook in registered_hooks:
        hook.remove()
    registered_hooks.clear()
    logging.debug("Deinstrumented model")
    return registered_hooks


def add_hook(layer, hook_collector, registered_hooks) -> list:
    """Adds a hook to a layer
        returns a list with the hook handle added to it.
        Callers to this method should pass in a list of previously
        registered hooks

    Args:
        layer (model layer): layer from the model
        hook_collector (Collector): hook collector
        registered_hooks (list): previously registered hooks

    Returns:
        list: appends to the list of previously registered hooks
    """
    hook_handle = layer.register_forward_hook(hook_collector)
    registered_hooks.append(hook_handle)
    return registered_hooks


def instrument_t5(cfg, model, hc: Collector) -> list:
    """Instruments a T5 model. It returns the list of hook handles
    that were added to the model. This list can be used to remove
    the instrumentation later.

    Args:
        cfg (dict): should contain the following configs:
        instrument_encoder, instrument_decoder, instrument_FF,
        instrument_attention
        model (HF model): HuggingFace model
        hc (Collector): Initialized Collector

    Returns:
        list: List of hook handles to use later
    """
    # Keep a list of these so we can de-instrument the model later
    registered_hooks = []
    if cfg["instrument_encoder"]:
        for block in model.encoder.block:
            for layer in block.layer:
                if isinstance(layer, T5LayerFF) and cfg["instrument_FF"]:
                    try:
                        add_hook(
                            layer.DenseReluDense.wi, hc, registered_hooks)
                        layer.DenseReluDense.wi.ns_name = "encoder." + \
                            block._get_name() + "." + layer._get_name() + \
                            ".DenseReluDense.wi"
                    except AttributeError:
                        add_hook(
                            layer.DenseReluDense.wi_0, hc, registered_hooks)
                        add_hook(
                            layer.DenseReluDense.wi_1, hc, registered_hooks)
                        layer.DenseReluDense.wi_0.ns_name = "encoder." + \
                            block._get_name() + "." + layer._get_name() + \
                            ".DenseReluDense.wi_0"
                        layer.DenseReluDense.wi_1.ns_name = "encoder." + \
                            block._get_name() + "." + layer._get_name() + \
                            ".DenseReluDense.wi_1"
                    add_hook(
                        layer.DenseReluDense.wo, hc, registered_hooks)
                    add_hook(
                        layer.DenseReluDense.act, hc, registered_hooks)
                    layer.DenseReluDense.wo.ns_name = "encoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".DenseReluDense.wo"
                    layer.DenseReluDense.act.ns_name = "encoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".DenseReluDense.act"
                if isinstance(layer, T5LayerSelfAttention) \
                        and cfg["instrument_attention"]:
                    add_hook(layer.SelfAttention.q, hc, registered_hooks)
                    add_hook(layer.SelfAttention.k, hc, registered_hooks)
                    add_hook(layer.SelfAttention.v, hc, registered_hooks)
                    add_hook(layer.SelfAttention.o, hc, registered_hooks)
                    add_hook(layer.layer_norm, hc, registered_hooks)
                    layer.SelfAttention.q.ns_name = "encoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".SelfAttention.q"
                    layer.SelfAttention.k.ns_name = "encoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".SelfAttention.k"
                    layer.SelfAttention.v.ns_name = "encoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".SelfAttention.v"
                    layer.SelfAttention.o.ns_name = "encoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".SelfAttention.o"
                    layer.layer_norm.ns_name = "encoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".layer_norm"
    if cfg["instrument_decoder"]:
        for block in model.decoder.block:
            for layer in block.layer:
                if isinstance(layer, T5LayerFF) and cfg["instrument_FF"]:
                    try:
                        add_hook(
                            layer.DenseReluDense.wi, hc, registered_hooks)
                        layer.DenseReluDense.wi.ns_name = "decoder." + \
                            block._get_name() + "." + layer._get_name() + \
                            ".DenseReluDense.wi"
                    except AttributeError:
                        add_hook(
                            layer.DenseReluDense.wi_0, hc, registered_hooks)
                        add_hook(
                            layer.DenseReluDense.wi_1, hc, registered_hooks)
                        layer.DenseReluDense.wi_0.ns_name = "decoder." + \
                            block._get_name() + "." + layer._get_name() + \
                            ".DenseReluDense.wi_0"
                        layer.DenseReluDense.wi_1.ns_name = "decoder." + \
                            block._get_name() + "." + layer._get_name() + \
                            ".DenseReluDense.wi_1"
                    add_hook(
                        layer.DenseReluDense.wo, hc, registered_hooks)
                    add_hook(
                        layer.DenseReluDense.act, hc, registered_hooks)
                    layer.DenseReluDense.wo.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".DenseReluDense.wo"
                    layer.DenseReluDense.act.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".DenseReluDense.act"
                if isinstance(layer, T5LayerSelfAttention) \
                        and cfg["instrument_attention"]:
                    add_hook(layer.SelfAttention.q, hc, registered_hooks)
                    add_hook(layer.SelfAttention.k, hc, registered_hooks)
                    add_hook(layer.SelfAttention.v, hc, registered_hooks)
                    add_hook(layer.SelfAttention.o, hc, registered_hooks)
                    add_hook(layer.layer_norm, hc, registered_hooks)
                    layer.SelfAttention.q.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".SelfAttention.q"
                    layer.SelfAttention.k.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".SelfAttention.k"
                    layer.SelfAttention.v.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".SelfAttention.v"
                    layer.SelfAttention.o.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".SelfAttention.o"
                    layer.layer_norm.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".layer_norm"
                if isinstance(layer, T5LayerCrossAttention) \
                        and cfg["instrument_attention"]:
                    add_hook(
                        layer.EncDecAttention.q, hc, registered_hooks)
                    add_hook(
                        layer.EncDecAttention.k, hc, registered_hooks)
                    add_hook(
                        layer.EncDecAttention.v, hc, registered_hooks)
                    add_hook(
                        layer.EncDecAttention.o, hc, registered_hooks)
                    add_hook(
                        layer.layer_norm, hc, registered_hooks)
                    layer.EncDecAttention.q.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".T5LayerCrossAttention.q"
                    layer.EncDecAttention.k.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".T5LayerCrossAttention.k"
                    layer.EncDecAttention.v.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".T5LayerCrossAttention.v"
                    layer.EncDecAttention.o.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".T5LayerCrossAttention.o"
                    layer.layer_norm.ns_name = "decoder." + \
                        block._get_name() + "." + layer._get_name() + \
                        ".layer_norm"
    add_hook(model.lm_head, hc, registered_hooks)
    model.lm_head.ns_name = "decoder.output"
    hc.last_layer = id(model.lm_head)
    return registered_hooks


def instrument_bert(cfg, model, hc: Collector) -> list:
    # Keep a list of these so we can de-instrument the model later
    registered_hooks = []
    if cfg["instrument_encoder"]:

        for i, lyr in enumerate(model.bert.encoder.layer):
            add_hook(
                lyr.attention.self.query, hc, registered_hooks)
            lyr.attention.self.query.ns_name = "encoder." + \
                f".self_attention.query_{i}"
            add_hook(
                lyr.attention.self.key, hc, registered_hooks)
            lyr.attention.self.key.ns_name = "encoder." + \
                f".self_attention.key_{i}"
            add_hook(
                lyr.attention.self.value, hc, registered_hooks)
            lyr.attention.self.value.ns_name = "encoder." + \
                f".self_attention.value_{i}"

            add_hook(
                lyr.attention.output.dense, hc, registered_hooks)
            lyr.attention.output.dense.ns_name = "encoder." + \
                f".self_attention.output.dense_{i}"

            add_hook(
                lyr.intermediate.dense, hc, registered_hooks)
            lyr.intermediate.dense.ns_name = "encoder." + \
                f".intermediate.dense_{i}"

            add_hook(
                lyr.output.dense, hc, registered_hooks)
            lyr.output.dense.ns_name = "encoder." + \
                f".output.dense_{i}"

    if cfg["instrument_decoder"]:

        add_hook(
            model.cls.predictions.transform.dense, hc, registered_hooks)
        model.cls.predictions.transform.dense.ns_name = "decoder." + \
            "cls.predictions.transform.dense"
        add_hook(
            model.cls.predictions.transform.LayerNorm, hc, registered_hooks)
        model.cls.predictions.transform.LayerNorm.ns_name = "decoder." + \
            "cls.predictions.transform.LayerNorm"

    add_hook(model.cls.predictions.decoder, hc, registered_hooks)
    model.cls.predictions.decoder.ns_name = "decoder.output"
    hc.last_layer = id(model.cls.predictions.decoder)
    return registered_hooks


def instrument_mpnet(cfg, model, hc: Collector) -> list:
    # Keep a list of these so we can de-instrument the model later
    registered_hooks = []
    if cfg["instrument_encoder"]:

        for i, lyr in enumerate(model.mpnet.encoder.layer):
            add_hook(
                lyr.attention.attn.q, hc, registered_hooks)
            lyr.attention.attn.q.ns_name = "encoder." + \
                f".attention.q_{i}"
            add_hook(
                lyr.attention.attn.k, hc, registered_hooks)
            lyr.attention.attn.k.ns_name = "encoder." + \
                f".attention.k_{i}"
            add_hook(
                lyr.attention.attn.v, hc, registered_hooks)
            lyr.attention.attn.v.ns_name = "encoder." + \
                f".attention.v_{i}"
            add_hook(
                lyr.attention.attn.o, hc, registered_hooks)
            lyr.attention.attn.o.ns_name = "encoder." + \
                f".attention.o_{i}"
            add_hook(
                lyr.attention.LayerNorm, hc, registered_hooks)
            lyr.attention.LayerNorm.ns_name = "encoder." + \
                f".attention.LayerNorm_{i}"

            add_hook(
                lyr.intermediate.dense, hc, registered_hooks)
            lyr.intermediate.dense.ns_name = "encoder." + \
                f".intermediate_dense_{i}"
            add_hook(
                lyr.intermediate.intermediate_act_fn, hc, registered_hooks)
            lyr.intermediate.intermediate_act_fn.ns_name = "encoder." + \
                f".intermediate_intermediate_act_fn_{i}"

            add_hook(
                lyr.output.dense, hc, registered_hooks)
            lyr.output.dense.ns_name = "encoder." + \
                f".output_dense_{i}"

    if cfg["instrument_decoder"]:

        add_hook(
            model.lm_head.dense, hc, registered_hooks)
        model.lm_head.dense.ns_name = "lm_head." + \
            f".dense_{i}"

    add_hook(model.lm_head.decoder, hc, registered_hooks)
    model.lm_head.decoder.ns_name = "decoder.output"
    hc.last_layer = id(model.lm_head.decoder)

    def generate_trampoline(input_ids):
        return model(input_ids)

    model.generate = generate_trampoline

    return registered_hooks


def instrument_llama2(cfg, model, hc: Collector) -> list:
    """Instruments a llama2 model. It returns the list of hook handles
    that were added to the model. This list can be used to remove
    the instrumentation later.

    Args:
        cfg (dict): should contain the following configs:
        instrument_encoder, instrument_decoder, instrument_FF,
        instrument_attention
        model (HF model): HuggingFace model
        hc (Collector): Initialized Collector

    Returns:
        list: List of hook handles to use later
    """
    # Keep a list of these so we can de-instrument the model later
    registered_hooks = []
    if cfg["instrument_embedding"]:
        add_hook(model.model.embed_tokens, hc, registered_hooks)
        model.model.embed_tokens.ns_name = "LlamaDecoder.embed_tokens"
    if cfg["instrument_decoder"]:
        for layer in model.model.layers:
            if cfg["instrument_attention"]:
                add_hook(
                    layer.self_attn.q_proj, hc, registered_hooks)
                layer.self_attn.q_proj.ns_name =\
                    "LlamaDecoder.self_attn.q_proj"
                add_hook(
                    layer.self_attn.k_proj, hc, registered_hooks)
                layer.self_attn.k_proj.ns_name =\
                    "LlamaDecoder.self_attn.k_proj"
                add_hook(
                    layer.self_attn.v_proj, hc, registered_hooks)
                layer.self_attn.v_proj.ns_name =\
                    "LlamaDecoder.self_attn.v_proj"
                add_hook(
                    layer.self_attn.o_proj, hc, registered_hooks)
                layer.self_attn.o_proj.ns_name =\
                    "LlamaDecoder.self_attn.o_proj"
            if cfg["instrument_FF"]:
                add_hook(
                    layer.mlp.gate_proj, hc, registered_hooks)
                layer.mlp.gate_proj.ns_name =\
                    "LlamaDecoder.mlp.gate_proj"
                add_hook(layer.mlp.up_proj, hc, registered_hooks)
                layer.mlp.up_proj.ns_name =\
                    "LlamaDecoder.mlp.up_proj"
                add_hook(
                    layer.mlp.down_proj, hc, registered_hooks)
                layer.mlp.down_proj.ns_name =\
                    "LlamaDecoder.mlp.down_proj"
                add_hook(
                    layer.mlp.act_fn, hc, registered_hooks)
                layer.mlp.act_fn.ns_name =\
                    "LlamaDecoder.mlp.act_fn"
    add_hook(model.model.norm, hc, registered_hooks)
    model.model.norm.ns_name = "LlamaDecoder.norm"
    add_hook(model.lm_head, hc, registered_hooks)
    model.lm_head.ns_name = "LlamaDecoder.output"
    hc.last_layer = id(model.lm_head)
    return registered_hooks


def instrument_mixtral_8x(cfg, model, hc: Collector) -> list:
    """Instruments a mistralai/Mixtral-8x7B-Instruct-v0.1 model.
    It returns the list of hook handles
    that were added to the model. This list can be used to remove
    the instrumentation later.

    Args:
        cfg (dict): should contain the following configs:
        instrument_encoder, instrument_decoder, instrument_FF,
        instrument_attention
        model (HF model): HuggingFace model
        hc (Collector): Initialized Collector

    Returns:
        list: List of hook handles to use later
    """
    # Keep a list of these so we can de-instrument the model later
    registered_hooks = []
    if cfg["instrument_embedding"]:
        add_hook(model.model.embed_tokens, hc, registered_hooks)
        model.model.embed_tokens.ns_name = "mixtral.embed_tokens"
    if cfg["instrument_decoder"]:
        for layer in model.model.layers:
            if cfg["instrument_attention"]:
                add_hook(
                    layer.self_attn.q_proj, hc, registered_hooks)
                layer.self_attn.q_proj.ns_name =\
                    "mixtral.self_attn.q_proj"
                add_hook(
                    layer.self_attn.k_proj, hc, registered_hooks)
                layer.self_attn.k_proj.ns_name =\
                    "mixtral.self_attn.k_proj"
                add_hook(
                    layer.self_attn.v_proj, hc, registered_hooks)
                layer.self_attn.v_proj.ns_name =\
                    "mixtral.self_attn.v_proj"
                add_hook(
                    layer.self_attn.o_proj, hc, registered_hooks)
                layer.self_attn.o_proj.ns_name =\
                    "mixtral.self_attn.o_proj"
                # add_hook(
                #
                #    layer.self_attn.rotary_emb, hc, registered_hooks)
                # layer.self_attn.rotary_emb.ns_name =\
                #    "mixtral.self_attn.rotary_emb"
            if cfg["instrument_FF"]:
                add_hook(
                    layer.block_sparse_moe.gate, hc, registered_hooks)
                layer.block_sparse_moe.gate.ns_name =\
                    "mixtral.block_sparse_moe.gate"
                for expert in layer.block_sparse_moe.experts:
                    add_hook(
                        expert.w1, hc, registered_hooks)
                    expert.w1.ns_name =\
                        "mixtral.expert.w1"
                    add_hook(
                        expert.w2, hc, registered_hooks)
                    expert.w2.ns_name =\
                        "mixtral.expert.w2"
                    add_hook(
                        expert.w3, hc, registered_hooks)
                    expert.w3.ns_name =\
                        "mixtral.expert.w3"
                    add_hook(
                        expert.act_fn, hc, registered_hooks)
                    expert.act_fn.ns_name =\
                        "mixtral.expert.act_fn_SiLU"

                add_hook(
                    layer.input_layernorm, hc, registered_hooks)
                layer.input_layernorm.ns_name =\
                    "mixtral.input_layernorm"
                add_hook(
                    layer.post_attention_layernorm, hc, registered_hooks)
                layer.post_attention_layernorm.ns_name =\
                    "mixtral.post_attention_layernorm"

    add_hook(model.model.norm, hc, registered_hooks)
    model.model.norm.ns_name = "mixtral.norm"
    add_hook(model.lm_head, hc, registered_hooks)
    model.lm_head.ns_name = "mixtral.lm_head"
    hc.last_layer = id(model.lm_head)
    return registered_hooks


def instrument_mistral_7b(cfg, model, hc: Collector) -> list:
    """Instruments a mistralai/Mixtral-8x7B-Instruct-v0.1 model.
    It returns the list of hook handles
    that were added to the model. This list can be used to remove
    the instrumentation later.

    Args:
        cfg (dict): should contain the following configs:
        instrument_encoder, instrument_decoder, instrument_FF,
        instrument_attention
        model (HF model): HuggingFace model
        hc (Collector): Initialized Collector

    Returns:
        list: List of hook handles to use later
    """
    # Keep a list of these so we can de-instrument the model later
    registered_hooks = []
    if cfg["instrument_embedding"]:
        add_hook(model.model.embed_tokens, hc, registered_hooks)
        model.model.embed_tokens.ns_name = "mistral.embed_tokens"
    if cfg["instrument_decoder"]:
        for layer in model.model.layers:
            if cfg["instrument_attention"]:
                add_hook(
                    layer.self_attn.q_proj, hc, registered_hooks)
                layer.self_attn.q_proj.ns_name =\
                    "mistral.self_attn.q_proj"
                add_hook(
                    layer.self_attn.k_proj, hc, registered_hooks)
                layer.self_attn.k_proj.ns_name =\
                    "mistral.self_attn.k_proj"
                add_hook(
                    layer.self_attn.v_proj, hc, registered_hooks)
                layer.self_attn.v_proj.ns_name =\
                    "mistral.self_attn.v_proj"
                add_hook(
                    layer.self_attn.o_proj, hc, registered_hooks)
                layer.self_attn.o_proj.ns_name =\
                    "mistral.self_attn.o_proj"
                # add_hook(
                #
                #    layer.self_attn.rotary_emb, hc, registered_hooks)
                # layer.self_attn.rotary_emb.ns_name =\
                #    "mistral.self_attn.rotary_emb"
            if cfg["instrument_FF"]:
                add_hook(
                    layer.mlp.gate_proj, hc, registered_hooks)
                layer.mlp.gate_proj.ns_name =\
                    "mistral.mlp.gate_proj"
                add_hook(
                    layer.mlp.up_proj, hc, registered_hooks)
                layer.mlp.up_proj.ns_name =\
                    "mistral.mlp.up_proj"
                add_hook(
                    layer.mlp.down_proj, hc, registered_hooks)
                layer.mlp.down_proj.ns_name =\
                    "mistral.mlp.down_proj"
                add_hook(
                    layer.mlp.act_fn, hc, registered_hooks)
                layer.mlp.act_fn.ns_name =\
                    "mistral.mlp.act_fn"

                add_hook(
                    layer.input_layernorm, hc, registered_hooks)
                layer.input_layernorm.ns_name =\
                    "mistral.input_layernorm"
                add_hook(
                    layer.post_attention_layernorm, hc, registered_hooks)
                layer.post_attention_layernorm.ns_name =\
                    "mistral.post_attention_layernorm"

    add_hook(model.model.norm, hc, registered_hooks)
    model.model.norm.ns_name = "mistral.norm"
    add_hook(model.lm_head, hc, registered_hooks)
    model.lm_head.ns_name = "mistral.lm_head"
    hc.last_layer = id(model.lm_head)
    return registered_hooks


def instrument_phi3(cfg, model, hc: Collector) -> list:
    """Instruments a Phi3 model. It returns the list of hook handles
    that were added to the model. This list can be used to remove
    the instrumentation later.

    Args:
        cfg (dict): should contain the following configs:
        instrument_encoder, instrument_decoder, instrument_FF,
        instrument_attention
        model (HF model): HuggingFace model
        hc (Collector): Initialized Collector

    Returns:
        list: List of hook handles to use later
    """
    # Keep a list of these so we can de-instrument the model later

    registered_hooks = []
    if cfg["instrument_embedding"]:
        add_hook(model.model.embed_tokens, hc, registered_hooks)
        model.model.embed_tokens.ns_name = "phi.embed_tokens"
        add_hook(model.model.embed_dropout, hc, registered_hooks)
        model.model.embed_dropout.ns_name = "phi.embed_tokens"
    if cfg["instrument_decoder"]:
        for layer in model.model.layers:
            if cfg["instrument_attention"]:
                add_hook(
                    layer.self_attn.o_proj, hc, registered_hooks)
                layer.self_attn.o_proj.ns_name =\
                    "phi3.self_attn.q_proj"
                add_hook(
                    layer.self_attn.qkv_proj, hc, registered_hooks)
                layer.self_attn.qkv_proj.ns_name =\
                    "phi3.self_attn.qkv_proj"
            if cfg["instrument_FF"]:
                add_hook(
                    layer.mlp.gate_up_proj, hc, registered_hooks)
                layer.mlp.gate_up_proj.ns_name =\
                    "phi3.mlp.gate_up_proj"
                add_hook(
                    layer.mlp.down_proj, hc, registered_hooks)
                layer.mlp.down_proj.ns_name =\
                    "phi3.mlp.down_proj"
                add_hook(
                    layer.mlp.activation_fn, hc, registered_hooks)
                layer.mlp.activation_fn.ns_name =\
                    "phi3.mlp.activation_fn"

                add_hook(
                    layer.input_layernorm, hc, registered_hooks)
                layer.input_layernorm.ns_name =\
                    "phi3.mlp.input_layernorm"
                add_hook(
                    layer.post_attention_layernorm, hc, registered_hooks)
                layer.post_attention_layernorm.ns_name =\
                    "phi3.mlp.post_attention_layernorm"

    add_hook(model.lm_head, hc, registered_hooks)
    model.lm_head.ns_name = "phi3.lm_head"
    hc.last_layer = id(model.lm_head)
    return registered_hooks
