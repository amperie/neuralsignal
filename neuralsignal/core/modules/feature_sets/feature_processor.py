import logging
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


# TODO: Redo this to use class definition

class FeatureProcessor:
    """Feature processing class to create features
    for training and at inference time. It takes the
    following inputs:
        - Feature config dict - dict of configs for
            each feature set. Format example:
            {
                "zone_feature_set": {
                    # processing_function is REQUIRED
                    # Signature is fn(scan, cfg_dict)
                    "processing_function": fn_instance,
                    "zone_size": 1024,
                    "use_full_zone_names": False}
            }
        - Scan data to build features from
    Processing function specs:
        - fn(scan, cfg_dict)
        cfg_dict is specific to the feature set and can be anything
        returns a dictionary that contains:
            - column_names: list of column names
            - column_values: list of column values
    """
    default_config = {
        "feature_sets": {},
    }

    def __init__(self, config: dict):
        self.config = config
        self.feature_sets = config["feature_configs"]
        self.scan = None

    def set_scan(self, scan):
        self.scan = scan

    def process_feature_set(self, feature_set_name: str):
        if feature_set_name not in self.feature_sets:
            raise ValueError(
                f"Feature set {feature_set_name} not found in config"
                f": {self.feature_configs}"
            )
        cfg = self.feature_configs[feature_set_name]
        fn = cfg['processing_function']
        retVal = fn(self.scan, cfg)
        retVal['feature_set_name'] = feature_set_name
        return retVal

    def process_all_feature_sets(self):
        retVal = {}
        for feature_set_name in self.feature_sets.keys():
            retVal[feature_set_name] = self.process_feature_set(
                feature_set_name
            )
        return retVal
