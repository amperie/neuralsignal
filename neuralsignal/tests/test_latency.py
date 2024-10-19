from neuralsignal.sdk.neuralsignal import SDK as ns_SDK
from neuralsignal.datasets.dataset_definitions import get_dataset
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.core.modules.detector import Detector
import time


# NeuralSignal SDK
ns_config = {
    "save_scans": False,
}

ns = ns_SDK(
    application_name="NS_Test",
    sub_application_name="latency",
    config=ns_config
)

cfg = {
    "application_name": "NS_Test",
    "sub_application_name": "latency",
    "dataset": "quora_duplicate_questions",
    "batch_size": 1,
    "row_limit": 1000,
    "detector_names": ["quora_duplicates"],
    "detectors": [],
    "indirect_instrumentation_config": {
        "collector_config": {
            "mode": "additive",
            "data_to_save": ["inputs", "outputs", "layer_info", "topology"],
            "zone_size": 1024,
            "zone_size_by_layer": {
                "default": 1024,
            },
            "layer_names_to_include": ['all'],
            "layer_indexes_to_include": [],
        }
    }
    }

ds = []
for d in cfg['detector_names']:
    detector = sdk_config.get_detector_config(d)
    if detector['enabled']:
        detector['application_name'] = cfg['application_name']
        detector['sub_application_name'] = cfg['sub_application_name']
        detector = Detector(detector)
        detector.enable_prediction = True
        ds.append(detector)
cfg['detectors'] = ds

batch = []

dataset = get_dataset(
    row_limit=cfg['row_limit'],
    dataset_name=cfg['dataset'])

total_latency = 0.0
rows_processed = 0
batch_size = cfg['batch_size']

for row in dataset:
    batch.append(row)
    if len(batch) == batch_size:
        start = time.time()
        res = ns.evaluate_indirect_output(batch, ds)
        end = time.time()
        total_latency += end - start
        rows_processed += len(batch)
        print(f"Elapsed time: {end - start}")
        print(f"Average per row: {(total_latency / rows_processed)}")
        batch = []
