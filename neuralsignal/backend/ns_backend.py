import logging
from neuralsignal.backend.mongo_backend import MongoBackend
from neuralsignal.backend.ns_be_impl_v1 import NSBackendImplV1
from neuralsignal.backend.file_backend import FileBackend
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.core.modules.s1_model import S1Model

logging.basicConfig(level=logging.INFO)


# This class abstracts the backend and provides a consistent
# interface for the SDK to interact with the backend.
# The specific implementation of the backend is connected
# as a plugin

class NSBackend:
    """
    This class abstracts the backend and provides a consistent
    interface for the SDK to interact with the backend.
    The specific implementation of the backend is connected
    as a plugin.
    Configuration options:
        - backend_type: noop, mongo, neuralsignal_v1
        - backend_config: configuration for the backend.
            Configuration specific to the backend type
        - backend_config should always contain:
            - application_name
            - sub_application_name
    """

    def __init__(self, config: dict = None) -> None:
        if "application_name" not in config:
            raise ValueError("Missing application_name in config")
        if "sub_application_name" not in config:
            raise ValueError("Missing sub_application_name in config")
        if "backend_config" not in config:
            be = sdk_config.get_backend_config()
            config['backend_config'] = be

        self.backend_type = config['backend_config']["backend_type"]
        self.backend_config = config['backend_config']
        self.backend_config['application_name'] =\
            config['application_name']
        self.backend_config['sub_application_name'] =\
            config['sub_application_name']
        if self.backend_type == "noop":
            self.backend = NoopBackend(self.backend_config)
        elif self.backend_type == "mongo":
            self.backend = MongoBackend(self.backend_config)
        elif self.backend_type == "neuralsignal_v1":
            self.backend = NSBackendImplV1(self.backend_config)
        elif self.backend_type == "file_backend":
            self.backend = FileBackend(self.backend_config)
        else:
            raise ValueError(f"Backend type {self.backend_type} not supported")

    def save_scan(self, scan) -> None:
        """Save a scan to the backend.
        scan is a GenerationInstance object"""
        return self.backend.save_scan(scan)

    def load_scan(self, scan_id: str):
        return self.backend.load_scan(scan_id)

    def deserialize_scan(self, doc):
        return self.backend.deserialize_scan(doc)

    def query(self, query: dict) -> list:
        return self.backend.query(query)

    def get_query_count(self, query: dict) -> int:
        return self.backend.get_query_count(query)

    def load_s1_model(self, model_id: str):
        return self.backend.load_s1_model(model_id)

    def save_s1_model(self, model: S1Model) -> str:
        return self.backend.save_s1_model(model)


class NoopBackend:
    def __init__(self, config: dict) -> None:
        pass

    def save_scan(self, scan) -> None:
        pass

    def load_scan(self, scan_id: str):
        pass

    def query(self, query: dict) -> list:
        pass

    def load_s1_model(self, model_id: str):
        pass

    def save_s1_model(self, model_id: str, model: S1Model) -> None:
        pass
