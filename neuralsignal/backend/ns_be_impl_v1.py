import logging
import mlflow
import os
import pickle
from neuralsignal.backend.mongo_backend import MongoBackend
from neuralsignal.core.modules.utils import string_to_filename
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


class NSBackendImplV1:

    """Implements the dev version of the NS backend.
    Mongo for storage, mlflow for models, elastic for vectors
    """
    def __init__(self, config: dict) -> None:
        self.config = config
        # Set the DB and Collection based on application_name
        config['db'] = config['application_name']
        config['col'] = config['sub_application_name']
        self.mng = MongoBackend(config)
        mlflow.set_tracking_uri(config['mlflow_uri'])

    # Interface methods
    def save_scan(self, scan) -> None:
        # TODO: functionality to save vector
        return self.mng.save_scan(scan)

    def load_scan(self, scan_id: str):
        return self.mng.load_scan(scan_id)

    def deserialize_scan(self, doc):
        return self.mng.deserialize_scan(doc)

    def query(self, query: dict) -> list:
        return self.mng.query(query)
    
    def get_query_count(self, query: dict) -> int:
        return self.mng.get_query_count(query)

    def load_s1_model(self, model_id: str):
        # Check to see if model is cached locally
        # If not, get it and cache it
        file_name = string_to_filename(model_id)
        file_name = f"{sdk_config.get('home')}/s1/{file_name}"
        exists = os.path.isfile(file_name)
        if exists:
            logging.info(
                f"Loading model {model_id} locally from {file_name}")
            return pickle.load(open(file_name, "rb"))
        else:
            logging.info(
                f"Model {model_id} not found locally. "
                "Downloading from backend."
            )
            model = mlflow.sklearn.load_model(model_id)
            with open(file_name, 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return model
