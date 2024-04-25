import logging
from neuralsignal.backend.mongo_backend import MongoBackend
from neuralsignal.backend.ns_be_impl_v1 import NSBackendImplV1

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
    """

    def __init__(self, config: dict = None) -> None:
        self.backend_type = config["backend_type"]
        self.backend_config = config
        if self.backend_type == "noop":
            self.backend = NoopBackend(self.backend_config)
        elif self.backend_type == "mongo":
            self.backend = MongoBackend(self.backend_config)
        elif self.backend_type == "neuralsignal_v1":
            self.backend = NSBackendImplV1(self.backend_config)
        else:
            raise ValueError(f"Backend type {self.backend_type} not supported")

    def save_scan(self, scan) -> None:
        """Save a scan to the backend.
        scan is a GenerationInstance object"""
        return self.backend.save_scan(scan)

    def load_scan(self, scan_id: str):
        return self.backend.load_scan(scan_id)

    def query(self, query: dict) -> list:
        return self.backend.query(query)

    def load_s1_model(self, model_id: str):
        return self.backend.load_s1_model(model_id)


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
