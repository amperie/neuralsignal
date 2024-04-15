import logging
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)


def load_model(model_config: dict) -> tuple[AutoTokenizer, AutoModel]:
    """Loads a model from a config

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
        "config: {model_config}")

    if model_config["quantization"] == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        model = AutoModel.from_pretrained(
            model_config["model_name"],
            device_map=model_config["device"],
            quantization_config=bnb_config)
        logging.info(
            f"Loaded {model_config['model_name']} in 4 bit quantization")

    elif model_config["quantization"] == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16
            )
        model = AutoModel.from_pretrained(
            model_config["model_name"],
            device_map=model_config["device"],
            quantization_config=bnb_config)
        logging.info(
            f"Loaded {model_config['model_name']} in 8 bit quantization")

    else:
        model = AutoModel.from_pretrained(
            model_config["model_name"],
            device_map=model_config["device"])
        logging.info(
            f"Loaded {model_config['model_name']} unquantized")

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    logging.info(
        f"Loaded {model_config['model_name']} tokenizer")

    return (tokenizer, model)
