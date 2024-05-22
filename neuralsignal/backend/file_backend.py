import logging
import os
import pickle
from neuralsignal.core.modules.utils import string_to_filename
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.core.modules.s1_model import S1Model
from neuralsignal.backend.backend_util import count_files_in_dir
from neuralsignal.core.modules.utils import generate_uuid

logging.basicConfig(level=sdk_config.logging_level())


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
        self.s1_home_dir =\
            f'{self.home_dir}s1/'

        if not os.path.exists(self.home_dir):
            os.makedirs(self.home_dir)
        if not os.path.exists(self.s1_home_dir):
            os.makedirs(self.s1_home_dir)

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

    def iterate_scans(self, query: dict, row_limit: int = 0):
        d = query['detector_name']
        scan_dir = f"{self.home_dir}/{d}/"
        scans = os.listdir(scan_dir)
        scans.sort()
        i = 0
        for scan in scans:
            if i > row_limit and row_limit != 0:
                break
            i += 1
            scan_id = scan.replace(".scan", "")
            retVal = self.load_scan(scan_id, d)

            yield retVal

    def get_scan_iterator_count(self, query: dict) -> int:
        d = query['detector_name']
        scan_dir = f"{self.home_dir}/{d}/"
        scans = os.listdir(scan_dir)
        return len(scans)

    def load_s1_model(self, model_id: str):
        file_name = self.s1_home_dir + string_to_filename(model_id)
        with open(file_name, 'rb') as handle:
            retVal = pickle.load(handle)
        return retVal

    def save_s1_model(self, model: S1Model) -> str:
        model_id =\
            f"{model.model_name}_{model.model_id}_{generate_uuid()}"
        model.set_id(model_id)
        file_name = self.s1_home_dir + string_to_filename(model_id)
        with open(file_name, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return model
