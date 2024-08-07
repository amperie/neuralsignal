
class FeatureSetBase:

    default_config = {
        # Options: "name_and_value_columns", "tensor_dict" and "pandas"
        "output_format": "name_and_value_columns",
    }

    def __init__(self, processing_function, config: dict):
        self.config = {**self.default_config, **config}
        self.processing_function = processing_function
        self.output_format = self.config["output_format"]

    def process_feature_set(self, scan: dict):
        raise NotImplementedError(
            "process_feature_set must be implemented in inherited classes"
            )
