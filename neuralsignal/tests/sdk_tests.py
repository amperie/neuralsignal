# from neuralsignal.sdk.neuralsignal import SDK
from neuralsignal.core.modules.model_instrumentation\
    import generate_from_batch
from neuralsignal.core.modules.model_instrumentation\
    import load_model
from neuralsignal.core.modules.detector import Detector


def test_generation():
    cfg = {
        "evaluation_mode": "qb",  # qb or direct_instrument
        "qb_config": {
            "qb_model": "t5-small",
            "zone_size:": 512,
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
    st = generate_from_batch(
        ["what do you think", "how about this?"], model, tokenizer,
        instrumentation_cfg={})

    print(st)


def test_detector_creation():
    cfg = {
        "S1_model": None,  # Either pass the model directly or specify its path
        "S1_model_path": "runs:/773318bc57c747e19cc0b8b5827809ee/xgboost1",  # If both are present S1_model is used
        "mlflow_uri": "http://z600.lan:8000",
        "prompt": "Test",
        "behavior_name": "test",
    }

    d = Detector(cfg)
    print(d)


test_detector_creation()
