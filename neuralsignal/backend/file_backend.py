import logging
import mlflow
import os
import pickle
from neuralsignal.core.modules.utils import string_to_filename
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.core.modules.s1_model import S1Model
from neuralsignal.backend.backend_util import count_files_in_dir
from neuralsignal.backend.backend_util import BackendQueryResults

logging.basicConfig(level=sdk_config.logging_level())


class FileQueryResults(BackendQueryResults):
    def __next__(self):
        return self.results.__next__()


class FileBackend:

    """Implements a filesystem backed backend.
    Required configs:
        application_name
        sub_application_name
    Optional configs:
        home: directory to store data in
    """
    def __init__(self, config: dict) -> None:
        self.config = config
        self.application_name = config['application_name']
        self.sub_application_name = config['sub_application_name']
        if 'home' in config:
            self.home_dir = config['home']
        else:
            self.home_dir = sdk_config.get('home')
        self.home_dir =\
            f'{self.home_dir}/FileBackend/{self.application_name}/'\
            f'{self.sub_application_name}/'

        if not os.path.exists(self.home_dir):
            os.makedirs(self.home_dir)

    # Interface methods
    def save_scan(self, scan) -> None:
        detection = list(scan.detections.keys())[0]
        target_dir = f"{self.home_dir}/{detection}"
        os.makedirs(target_dir, exist_ok=True)
        ct = count_files_in_dir(target_dir, ".scan")
        fp = f"{target_dir}/{ct}.scan"
        with open(fp, 'wb') as handle:
            pickle.dump(scan, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_scan(self, scan_id: str, detection: str):
        fp = f"{self.home_dir}/{detection}/{scan_id}.scan"
        with open(fp, 'rb') as handle:
            retVal = pickle.load(handle)
        return retVal

    def deserialize_scan(self, doc):
        pass

    def query(self, query: dict) -> list:
        return self.mng.query(query)

    def get_query_count(self, query: dict) -> int:
        return self.mng.get_query_count(query)

    def iterate_scans(self, query: dict, row_limit: int):
        d = query['detector_name']
        scan_dir = f"{self.home_dir}/{d}/"
        scans = os.listdir(scan_dir)
        i = 0
        for scan in scans:
            if i > row_limit:
                break
            i += 1
            yield self.load_scan(scan, d)

    def load_s1_model(self, model_id: str):
        # TODO: load an S1Model object, not just the model itself

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

    def save_s1_model(self, model: S1Model) -> str:
        experiment_name =\
            f"{self.config['application_name']}_"\
            f"{self.config['sub_application_name']}"
        model.config["mlflow_info"] = save_to_mlflow(
            model, self.config["mlflow_uri"], experiment_name)
        model.config['model_id'] = model.config['mlflow_info']._model_uri
        model.set_id(model.config['mlflow_info']._model_uri)
        return model
