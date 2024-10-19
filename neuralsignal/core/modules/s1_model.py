import logging
from neuralsignal.core.modules.feature_sets.feature_set_factory\
    import make_feature_set
from neuralsignal.core.modules.feature_sets.feature_processor\
    import FeatureProcessor
from neuralsignal.core.modules.utils import mlflow_str_to_dict
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


class S1Model:
    """
    Simple class to just wrap a model so we can include metadata
    about it an define interfaces for using it
    """

    default_config = {
        "application_name": None,
        "sub_application_name": None,
        "model_name": None,
        "model_id": None,
        "dataset_path": None,
        "optimization_metric": None,
        'metrics': {},
        'params': {},
        'metadata': {},
        'tags': {},
        'artifacts': {},
        'figures': {},
        'description': "",
    }

    def __init__(self, config: dict) -> None:

        self.config = {**self.default_config, **config}
        self.model_name = config["model_name"]
        self.model_id = config["model_id"]
        self.model = config['model']

    def init_feature_sets(self):
        fsl = self.get_metadata_item('data')['params']['feature_set_list']
        # Convert from string to list
        fsl = mlflow_str_to_dict(fsl)
        fss = []
        for fs in fsl:
            fsc = self.get_metadata_item('data')\
                ['params'][f'feature_set_{fs}']
            fsc = mlflow_str_to_dict(fsc)
            fss.append(make_feature_set(fs, fsc))
        self.fp = FeatureProcessor(fss)

    def set_id(self, model_id: str):
        self.model_id = model_id

    def set_metadata(self, metadata: dict):
        self.metadata = metadata
        self.init_feature_sets()

    def get_metadata_item(self, name: str):
        return self.metadata[name]

    def __getitem__(self, name: str):
        return self.config[name]

    def predict(self, data, featurize=False):
        if featurize:
            inf_data = self.fp.featurize_scan(data)
        else:
            inf_data = data
        return self.model.predict(inf_data)

    def predict_proba(self, data, featurize=False):
        if featurize:
            inf_data = self.fp.featurize_scan(data)
        else:
            inf_data = data
        return self.model.predict_proba([inf_data[1]])
