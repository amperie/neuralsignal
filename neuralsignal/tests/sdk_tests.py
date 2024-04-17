from neuralsignal.sdk.neuralsignal import SDK
from neuralsignal.core.modules.model_instrumentation import generate_from_string
from neuralsignal.core.modules.model_instrumentation import load_model

cfg = {
    "evaluation_mode": "qb",  # qb or direct_instrument
    "qb_config": {
        "qb_model": "t5-small",
        "zone_size:": 1024,
        "qb_batch_size": 1,
        "quantization": "no_quantization",
        "device": "cpu",
    },
    "save_scans": False,  # Save scans to backend
    "backend_config": {},  # Backend endpoint
    "S1_model": None,  # S1 model can't be None
}

# s = SDK(cfg)
# s.evaluate_single_output("Hello, world!")

tokenizer, model = load_model({
    "model_name": "t5-small", "device": "cpu",
    "quantization": "no_quantization"
    })
st = generate_from_string("what do you think of this?", model, tokenizer)

print(st)
