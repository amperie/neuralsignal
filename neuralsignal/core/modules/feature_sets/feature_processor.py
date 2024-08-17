import logging
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.core.modules.feature_sets.feature_set_base\
    import FeatureSetBase

logging.basicConfig(level=sdk_config.logging_level())


# TODO: Redo this to use class definition

class FeatureProcessor:

    def __init__(self, feature_sets: list = []):
        self.feature_sets = feature_sets

    def add_feature_set(self, feature_set):
        self.feature_sets.append(feature_set)

    def set_scan(self, scan):
        self.scan = scan

    def process_feature_set(
            self, feature_set: FeatureSetBase,
            output_format="name_and_value_columns"
            ):

        original_output_format = feature_set.config["output_format"]
        feature_set.config["output_format"] = output_format
        retVal = feature_set.process_feature_set(self.scan)
        feature_set.config["output_format"] = original_output_format
        return retVal

    def process_all_feature_sets(self, output_format="name_and_value_columns"):
        retVal = {}
        for feature_set in self.feature_sets:
            key = feature_set.get_feature_set_name()
            retVal[key] = self.process_feature_set(feature_set, output_format)

        return retVal

    def featurize(self, output_format="name_and_value_columns"):
        processed = self.process_all_feature_sets(
            output_format="name_and_value_columns"
            )

        cols = []
        vals = []
        for fs in processed.keys():
            cols = cols + processed[fs][0]
            vals = vals + processed[fs][1]
        return (cols, vals)
