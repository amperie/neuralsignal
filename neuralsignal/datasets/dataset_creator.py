import logging
import os
import pandas as pd
from neuralsignal.backend.ns_backend import NSBackend
from neuralsignal.core.modules.tensors import featurize_tensor_dict
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


class DatasetCreator:
    """Generates an S1 dataset from a set of scans
    Usage:
            dc = DatasetCreator({
                "application_name": "<required>",
                "sub_application_name": "<required>",
                "zone_size": 1024,
                "row_limit": 0,
                "detector_name": None,  # * for all/any
                "write_header": True,
                "use_full_zone_names": False,
                "include_output": False,
                "write_to_file": True,
                "file_out": None,
                "build_in_memory": True,
                "use_gt_as_target": True,
                "tensor_field_to_use": "outputs",
                })
            d = dc.create_dataset(
                {query in the application and sub_application space}
            )
    """

    default_config = {
        "zone_size": 1024,
        "row_limit": 0,
        "detector_name": None,  # * for all/any
        "write_header": True,
        "use_full_zone_names": False,
        "include_output": False,
        "write_to_file": True,
        "file_out": None,
        "build_in_memory": True,
        "use_gt_as_target": True,
        "tensor_field_to_use": "outputs",
        "overwrite_dataset_file": True,
    }

    def __init__(self, config: dict):
        if "application_name" not in config:
            raise ValueError("Missing application_name in config")
        if "sub_application_name" not in config:
            raise ValueError("Missing sub_application_name in config")
        if "detector_name" not in config:
            raise ValueError("Missing detector_name in config")
        self.config = {**self.default_config, **config}
        if self.config["file_out"] is None and self.config["write_to_file"]:
            raise ValueError("Missing file_out in config")
        self.be = NSBackend(config)

    def create_dataset(self, query: dict):
        """
        Creates an S1 dataset from a query.
        Returns a tuple:
        retVal[0] = path to file out if write_to_file is True
        retVal[1] = pandas dataframe if build_in_memory is True
        Exception is raised if both these configs are set to False
        """
        if not self.config["write_to_file"] and\
                not self.config["build_in_memory"]:
            raise ValueError(
                "Specify either write_to_file or build_in_memory or both")
        logging.info(
            f"Building dataset with query: {query} "
            f"row_limit: {self.config['row_limit']}"
            )

        if self.config["write_to_file"] and\
                self.config["overwrite_dataset_file"] and\
                os.path.isfile(self.config["file_out"]):
            logging.info(
                f"Overwriting dataset: {self.config['file_out']}")
            os.remove(self.config["file_out"])

        # Setup the query with the detection
        if self.config["detector_name"] != "*":
            query[f"detections.{self.config['detector_name']}"] =\
                {"$ne": None}
        # Query for the documents to use for the dataset
        if self.config["row_limit"] == 0:
            mng_cursor = self.be.query(query)
        else:
            mng_cursor = self.be.query(query).limit(
                        self.config["row_limit"])

        # Get the doc count of the query
        doc_count = self.be.get_query_count(query)
        logging.info(f"Processing {doc_count} documents from query: {query}")

        # Setup for a couple different ways that the dataset
        # can be built. To file or in memory or both
        if self.config["write_to_file"]:
            f = open(self.config["file_out"], 'a')

        if self.config["build_in_memory"]:
            data = []

        # Build the header if needed by getting the first doc
        # to figure out how many zones there are
        header = "target,"
        if self.config['include_output']:
            header += "out,"
        if (self.config["write_header"] and self.config["write_to_file"])\
                or self.config["build_in_memory"]:
            first = self.be.deserialize_scan(mng_cursor.next())
            logging.info(f"Processing header and first doc {first['_id']}")
            t = featurize_tensor_dict(
                first['outputs'], self.config["zone_size"],
                first['zone_size'], first['layer_id_to_name']
            )
            for i in range(0, len(t[1])):
                if self.config["use_full_zone_names"]:
                    header += f"{t[2][i]},"
                else:
                    header += f"{t[1][i]},"
            f.write(f'{header}\n')

            # Write values for the first doc and in memory
            if self.config["write_to_file"]:
                row = ','.join(map(str, t[0])) + "\n"
                if self.config['include_output']:
                    row = first['output'].replace(",", " ") + ',' + row
                row = str(int(first['ground_truth'])) + ',' + row
                f.write(row)
            if self.config["build_in_memory"]:
                row = t[0]
                if self.config['include_output']:
                    row = [first['output'].replace(",", " ")] + row
                row = [int(first['ground_truth'])] + row
                data.append(row)

        # Process the rest of the docs
        iteration = 1
        for doc in mng_cursor:
            if iteration % 100 == 0:
                logging.info(f"Iteration {iteration} of {doc_count}")
            doc = self.be.deserialize_scan(doc)
            t = featurize_tensor_dict(
                doc['outputs'], self.config["zone_size"],
                doc['zone_size'], doc['layer_id_to_name']
            )
            if self.config["write_to_file"]:
                row = ','.join(map(str, t[0])) + "\n"
                if self.config['include_output']:
                    row = doc['output'].replace(",", "") + ',' + row
                row = str(int(doc['ground_truth'])) + ',' + row
                f.write(row)
            if self.config["build_in_memory"]:
                row = t[0]
                if self.config['include_output']:
                    row = [doc['output'].replace(",", "")] + row
                row = [int(doc['ground_truth'])] + row
                data.append(row)
            iteration += 1
        file_retVal = None
        memory_retVal = None

        if self.config["write_to_file"]:
            f.flush()
            f.close()
            file_retVal = self.config["file_out"]

        if self.config["build_in_memory"]:
            memory_retVal =\
                pd.DataFrame(data, columns=header.split(",")[:-1])

        logging.info("Done processing")
        return (file_retVal, memory_retVal)
