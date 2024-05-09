from neuralsignal.backend.ns_backend import NSBackend


be = NSBackend({
    "application_name": "sdk_diagram",
    "sub_application_name": "hivemapper"
})

ds = be.query({})

for doc in ds:
    s = be.deserialize_scan(doc)
    print(s)


class DatasetCreator:
    """Generates an S1 dataset from a set of scans
    """

    default_config = {
        "zone_size": 1024,
        "row_limit": 0,
        "write_to_file": True,
        "file_out": None,
        "build_in_memory": False,
        "use_gt_as_target": True,
    }

    def __init__(self, config: dict):
        if "application_name" not in config:
            raise ValueError("Missing application_name in config")
        if "sub_application_name" not in config:
            raise ValueError("Missing sub_application_name in config")
        self.config = {**self.default_config, **config}
        self.be = NSBackend(config)
