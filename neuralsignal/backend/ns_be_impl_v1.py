import logging
import mlflow
from neuralsignal.backend.mongo_backend import MongoBackend
from neuralsignal.core.modules.generation_instance import GenerationInstance

logging.basicConfig(level=logging.INFO)


class NSBackendImplV1:

    """Implements the dev version of the NS backend.
    Mongo for storage, mlflow for models
    """
    def __init__(self, config: dict) -> None:
        self.config = config
        self.mng = MongoBackend(config)
        mlflow.set_tracking_uri(config['mlflow_uri'])

    # Interface methods
    def save_scan(self, scan: GenerationInstance) -> None:
        return self.mng.write_dict_to_mongo(scan.data)

    def load_scan(self, scan_id: str) -> GenerationInstance:
        return self.mng.read_dict_from_mongo(scan_id)

    def query(self, query: dict) -> list:
        return self.mng.query(query)

    def load_s1_model(self, model_id: str):
        return mlflow.sklearn.load_model(model_id)
