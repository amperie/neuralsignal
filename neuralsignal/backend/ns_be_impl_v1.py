import logging
import mlflow
import os
import pickle
from neuralsignal.backend.mongo_backend import MongoBackend
from neuralsignal.core.modules.utils import string_to_filename
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.core.modules.s1_model import S1Model
from neuralsignal.backend.backend_util import save_to_mlflow

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
        self.mlflow_uri = config['mlflow_uri']
        self.mlflow_register_model = config['mlflow_register_model']
        if self.mlflow_uri == "databricks":
            self.mlflow_experiment_path = config['mlflow_experiment_path']
            os.environ['DATABRICKS_HOST'] = config['DATABRICKS_HOST']
            os.environ['DATABRICKS_TOKEN'] = config['DATABRICKS_TOKEN']

    # Interface methods
    def save_scan(self, scan) -> None:
        # TODO: functionality to save vector
        return self.mng.save_scan(scan)

    def load_scan(self, scan_id: str, detection: str = None):
        return self.mng.load_scan(scan_id)

    def deserialize_scan(self, doc):
        return self.mng.deserialize_scan(doc)

    def query(self, query: dict) -> list:
        return self.mng.query(query)

    def get_query_count(self, query: dict) -> int:
        return self.mng.get_query_count(query)

    def iterate_scans(self, query: dict, row_limit: int = 0):
        return self.mng.iterate_scans(query, row_limit)

    def get_scan_iterator_count(self, query: dict) -> int:
        return self.mng.get_scan_iterator_count(query)

    def load_s1_model(self, model_id: str):

        # Check to see if model is cached locally
        # If not, get it and cache it
        file_name = string_to_filename(model_id)
        file_name = f"{sdk_config.get('home')}/s1/{file_name}"
        exists = os.path.isfile(file_name)
        if exists:
            logging.info(
                f"Loading model {model_id} locally from {file_name}")
            model = pickle.load(open(file_name, "rb"))
        else:
            logging.info(
                f"Model {model_id} not found locally. "
                "Downloading from backend."
            )
            model = mlflow.sklearn.load_model(model_id)
            with open(file_name, 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        cfg = {
            'model_id': model_id,
            'model': model,
            'application_name': self.config['application_name'],
            'sub_application_name': self.config['sub_application_name'],
            'model_name': model_id
        }
        retVal = S1Model(cfg)
        return retVal

    def save_s1_model(self, model: S1Model):
        """
        Saves an S1Model object by updating its mlflow
        information and model_id.

        Parameters:
            model (S1Model): The S1Model object to be saved.

        Returns:
            S1Model: The saved S1Model object.
        """
        experiment_name =\
            f"{self.config['application_name']}"

        if self.mlflow_uri == "databricks":
            experiment_name =\
                f"{self.config['mlflow_experiment_path']}/{experiment_name}"

        if "run_name" in model.config:
            run_name = model.config["run_name"]
        else:
            run_name =\
                f"{self.config['sub_application_name']}"
        model.config["mlflow_info"] = save_to_mlflow(
            model, self.config["mlflow_uri"], experiment_name, run_name,
            self.mlflow_register_model)

        if self.mlflow_register_model:
            model.config['model_id'] = model.config['mlflow_info']._model_uri
            model.set_id(model.config['mlflow_info']._model_uri)

        return model
