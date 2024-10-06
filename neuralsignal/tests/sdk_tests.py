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
from neuralsignal.core.modules.feature_sets.feature_set_zones\
    import FeatureSetZones
from neuralsignal.core.modules.feature_sets.feature_set_tuned_lens\
    import FeatureSetTunedLens
from neuralsignal.core.modules.feature_sets.feature_processor\
    import FeatureProcessor
from neuralsignal.automation.dataset_automation_core import run_experiment
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

    d = sdk_config.get_detector_config("quora_duplicates")
    d['application_name'] = "sdk_testing"
    d['sub_application_name'] = "quora"
    d = Detector(d)

    cfg = {
        "dataset": "quora_duplicate_questions",
        "detectors": [d],
        "application_name": "sdk_testing",
        "sub_application_name": "quora",
        "max_new_tokens": 1,
        "row_limit": 0,
        "preprocess_dataset": False,
        "preprocess_params": {
            "source_ds": "DROP",
            "rows": 40
        },
        "indirect_instrumentation_config": {
            "collector_config": {
                "zone_size": 64,
                "zone_size_by_layer": {
                    "SelfAttention.o": 1,
                    "SelfAttention.q": 4
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
        "dataset_path": "/Users/pablo/Code/neuralsignal/halubench_t5-small_hallucination.csv",
        # "dataset_path": "H:\\My Drive\\Code\\NeuralSignal\\halubench_t5-small_hallucination.csv",
        "description": "testing",
        "tags": {"testtag": "testtag"},
        "metadata": {"testmd": "testmd"},
        "row_limit": 1930,
        "create_reduced_feature_model": True
    }

    mt = S1Trainer(cfg)
    m = mt.train_model()

    be = NSBackend(cfg)
    model = be.load_s1_model(m.model_id)
    print(model)


def load_scan(scan_id: str, detection: str = "hallucination"):
    cfg = {
        "application_name": "sdk_testing_zs_1",
        "sub_application_name": "hb_drop",
    }
    be = NSBackend(cfg)
    scan = be.load_scan(scan_id, detection)
    return scan


def test_feature_processor():
    # from neuralsignal.core.modules.feature_sets.feature_set_zones\
    #    import FeatureSetZones
    from neuralsignal.core.modules.feature_sets.feature_processor\
        import FeatureProcessor
    from neuralsignal.core.modules.feature_sets.feature_set_t_f_diff\
        import FeatureSetTrueFalseDiff

    scan = load_scan("66caa52bd6b651131619e073", "hallucination")
    scan = load_scan("66caa52ed6b651131619e307", "hallucination")
    scan = load_scan("66caa533d6b651131619e7dd", "hallucination")
    #scan = load_scan("66caa539d6b651131619ecb7", "hallucination")
    #scan = load_scan("66caa53dd6b651131619f04f", "hallucination")

    scan = load_scan("66caa52bd6b651131619e073", "hallucination")
    scan = load_scan("66caa52ed6b651131619e307", "hallucination")
    scan = load_scan("66caa533d6b651131619e7dd", "hallucination")
    """zones_cfg = {
        "name": "zones",
        "target_zone_size": {"default": 256, ".o": 512, ".q": 1024},
        "field_to_process": "outputs",
        "layer_names_to_include": [".o", ".q"],
        "layer_indexes_to_include": [],
        "output_format": "pandas",
    }"""
    # fsz = make_feature_set("zones", cfg)
    # fsz = FeatureSetZones(cfg)
    cfg = {
        "model_name": "google/flan-t5-large",
        "hf_token": "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",
        "layers_to_process": [".wo"],
    }

    tf = FeatureSetTrueFalseDiff(cfg)

    fp = FeatureProcessor(feature_sets=[tf])
    fp.set_scan(scan)
    retVal = fp.process_all_feature_sets()
    print(retVal)


def test_ds_create_feature_processor():
    cfg = {
        "target_zone_size": {"default": 512, ".o": 1024, ".q": 2048},
        "field_to_process": "outputs",
        "layer_names_to_include": [".o", ".q", "act"],
        "layer_indexes_to_include": [],
        "output_format": "pandas",
    }
    fsz = FeatureSetZones(cfg)
    fp = FeatureProcessor([fsz])

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
        "feature_processor": fp,
        }

    dc = DatasetCreator(cfg)
    retVal = dc.create_dataset(cfg['query'])
    return retVal


def test_run_experiment():
    cfg = {
        "target_zone_size": {"default": 512, ".o": 1024, ".q": 2048},
        "field_to_process": "outputs",
        "layer_names_to_include": [".o", ".q", "act"],
        "layer_indexes_to_include": [],
        "output_format": "pandas",
    }
    fsz = FeatureSetZones(cfg)
    fp = FeatureProcessor([fsz])
    cfg = {
        "detector_names": ['halu_prompt4_oneshot'],
        "scan_cache_directory": "J:\\Temp\\scan_cache",
        "feature_processor": fp,
    }
    run_experiment(cfg)


def test_tunedlens():

    cfg = {
        "application_name": "sdk_testing_zs_1",
        "sub_application_name": "hb_drop",
    }
    be = NSBackend(cfg)
    s = be.iterate_scans(
        {"detector_name": "halu_prompt4_oneshot"}, row_limit=10)

    tl_cfg = {
        "model_name": "google/flan-t5-large",
        "hf_token": "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",
        "layers_to_process": [".wo"],
    }
    fstl = FeatureSetTunedLens(tl_cfg)

    # nsm = fstl.load_model("TL-NN")

    fstl.process_training_data(s, skip_layers=40)
    train_cfg = {
        "logits_dim": 32128,
        "hidden_layer_dim": 10000,
        "training_split": 0.66,
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.00000005,
        "model_name": "TL-NN"
    }
    fstl.train_feature_set(train_cfg)

    print()


# test_tunedlens()
test_s1_model()
# load_scan("66ad0140b2d466a1795d91a3", "hallucination")
# test_detector_creation()
# test_generation()
# test_ds_create_feature_processor()
# test_run_experiment()
# test_feature_processor()
test_ds_runner()
# test_ds_create()
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
