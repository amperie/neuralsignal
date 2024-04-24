from neuralsignal.sdk.neuralsignal import SDK
from neuralsignal.core.modules.model_instrumentation\
    import generate_from_batch
from neuralsignal.core.modules.model_instrumentation\
    import load_model
from neuralsignal.core.modules.detector import Detector


def test_generation():

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
        "S1_model": None,
        # Either pass the model directly or specify its path
        "S1_model_path": "runs:/773318bc57c747e19cc0b8b5827809ee/xgboost1",
        # If both are present S1_model is used
        "mlflow_uri": "http://z600.lan:8000",
        "prompt": "Test",
        "behavior_name": "test",
    }

    d = Detector(cfg)
    print(d)


def test_sdk():

    s = SDK()
    eval = {
        "input": "input1",
        "context": "context1",
        "output": "output1",
        "metadata": "metadata1",
    }
    eval2 = {
        "input": "input2",
        "context": "context2",
        "output": "output2",
        "metadata": "metadata2",
    }
    ins = [eval, eval2]

    d1 = {
        "S1_model": None,
        # Either pass the model directly or specify its path
        "S1_model_path": "runs:/773318bc57c747e19cc0b8b5827809ee/xgboost1",
        # If both are present S1_model is used
        "prompt": "testing hallucination prompt {input} thanks {context}",
        "behavior_name": "hallucination",
        "enabled": "True",
    }
    d1 = Detector(d1)
    d2 = {
        "S1_model": None,
        # Either pass the model directly or specify its path
        "S1_model_path": "runs:/773318bc57c747e19cc0b8b5827809ee/xgboost1",
        # If both are present S1_model is used
        "prompt": "testing bias prompt {input} thanks {output}",
        "behavior_name": "toxicity",
        "enabled": "True",
    }
    d2 = Detector(d2)

    gis = s.evaluate_indirect_output(ins, [d1, d2])

    print(gis)


# test_detector_creation()
# test_generation()
test_sdk()
