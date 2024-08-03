from neuralsignal.sdk.neuralsignal import SDK
from neuralsignal.core.modules.model_instrumentation\
    import generate_from_batch
from neuralsignal.core.modules.model_instrumentation\
    import load_model
from neuralsignal.core.modules.detector import Detector
from neuralsignal.datasets.dataset_runner import DatasetRunner
from neuralsignal.datasets.dataset_creator import DatasetCreator
from neuralsignal.datasets.s1_trainer import S1Trainer
from neuralsignal.backend.ns_backend import NSBackend
from neuralsignal.core.modules.neuralsignal_config import sdk_config


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

    s = SDK(
        application_name="demo",
        sub_application_name="demo",
    )
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
    # d1 = Detector(d1)
    d2 = {
        "S1_model": None,
        # Either pass the model directly or specify its path
        "S1_model_path": "runs:/773318bc57c747e19cc0b8b5827809ee/xgboost1",
        # If both are present S1_model is used
        "prompt": "testing bias prompt {input} thanks {output}",
        "behavior_name": "toxicity",
        "enabled": "True",
    }
    # d2 = Detector(d2)

    gis = s.evaluate_indirect(ins, ['hallucination', 'text_categorizer'])
    gis = s.evaluate_indirect_output(ins, [d1, d2])

    print(gis)


def test_ds_runner():

    d = sdk_config.get_detector_config("hallucination")
    d['application_name'] = "sdk_testing"
    d['sub_application_name'] = "zones_size_test"
    d = Detector(d)

    cfg = {
        "dataset": "HaluBench",
        "detectors": [d],
        "application_name": "sdk_testing",
        "sub_application_name": "zones_size_test",
        "max_new_tokens": 1,
        "row_limit": 0,
        "preprocess_dataset": True,
        "preprocess_params": {
            "source_ds": "DROP",
            "rows": 40
        },
        "indirect_instrumentation_config": {
            "collector_config": {
                "zone_size": 64,
                "zone_size_by_layer": {
                    "SelfAttention.o": 1,
                    "SelfAttention.q": 256
                },
                "layer_names_to_include": [
                    "SelfAttention.o", "SelfAttention.q", "norm"],
                "layer_indexes_to_include": [2],
            }
        }

    }
    dsr = DatasetRunner(cfg)
    dsr.run()

    print("")


def test_ds_create():

    cfg = {
        "application_name": "sdk_attention_test",
        "sub_application_name": "squad_v2_right_wrong_pairs",
        "row_limit": 20,
        "write_to_file": True,
        "build_in_memory": True,
        "file_out": "J:\\Temp\\test.csv",
        "detector_name": "hallucination",
        "query": {},
        # "query": {'$and': [{'metadata.row': {'$gt': 200}},
        # "query": {'$and': [{'metadata.row': {'$gt': 200}}, 
        # {'metadata.type': {'$ne': 'qa_rewrite_wrong'}}]},
        "zone_size": 1024,
        "use_full_zone_names": True,
        "use_gt_as_target": True,
        "passthrough_fields": ['zone_size'],
        "featurize_delta_layers": [],
        "featurize_delta_by_layer_name": [],
        "featurize_zones_data": False,
        "featurize_layer_distributions_layers": [],
        "featurize_layer_distributions_bin_count": 10,
        "featurize_embedding_vector_layers": ['SelfAttention.o'],
        "featurize_embedding_vector_range": -1,
        "featurize_embedding_vector_mode": "delta",
        }

    dc = DatasetCreator(cfg)
    retVal = dc.create_dataset(cfg['query'])
    return retVal


def test_s1_model():
    cfg = {
        "application_name": "sdk_squad_v2",
        "sub_application_name": "data",
        "model_name": "testing-s1",
        "dataset_path": "test.csv",
        "description": "testing",
        "tags": {"testtag": "testtag"},
        "metadata": {"testmd": "testmd"},
        "row_limit": 1930,
        "backend_config": {
            "backend_type": "file_backend",
        }
    }

    mt = S1Trainer(cfg)
    m = mt.train_model()

    be = NSBackend(cfg)
    model = be.load_s1_model(m.model_id)
    print(model)


def load_scan(scan_id: str, detection: str = "hallucination"):
    cfg = {
        "application_name": "sdk_testing",
        "sub_application_name": "zones_size_test",
    }
    be = NSBackend(cfg)
    scan = be.load_scan(scan_id, detection)
    return scan


def test_dataset_load():
    pass


load_scan("66ad0140b2d466a1795d91a3", "hallucination")
# test_detector_creation()
# test_generation()
# test_sdk()
test_ds_runner()
test_ds_create()
# test_s1_model()


def test_backend():
    cfg = {
        "application_name": "sdk_squad_v2",
        "sub_application_name": "data",
        "model_name": "testing-s1",
        "dataset_path": "squad.csv",
        "description": "testing",
        "tags": {"testtag": "testtag"},
        "metadata": {"testmd": "testmd"},
        "row_limit": 1930,
        "backend_config": {
            "backend_type": "file_backend",
        }
    }

    be = NSBackend(cfg)
    s = be.load_scan("0", "hallucination")
    s = be.iterate_scans({"detector_name": "hallucination"}, 10)
    # print(f"{s}____{id(s)}____{s.data['generation_correlation_id']}")
    for s in be.iterate_scans({"detector_name": "hallucination"}, 10):
        print(f"{s}____{id(s)}____{s.data['generation_correlation_id']}")

    print(be.get_scan_iterator_count({"detector_name": "hallucination"}))
    print(s)


# test_backend()
