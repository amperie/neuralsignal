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
        "passthrough_fields": [],
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
            query['detector_name'] =\
                self.config['detector_name']
        # Query for the documents to use for the dataset
        cursor = self.be.iterate_scans(
            query, row_limit=self.config["row_limit"])

        # Get the doc count of the query
        doc_count = self.be.get_scan_iterator_count(query)
        logging.info(
            f"Processing {min(doc_count, self.config['row_limit'])} "
            f"of {doc_count} documents from query: {query}")

        # Setup for a couple different ways that the dataset
        # can be built. To file or in memory or both
        if self.config["write_to_file"]:
            f = open(self.config["file_out"], 'a')

        if self.config["build_in_memory"]:
            data = []

        # Iterate over the scans but build header first
        # if required

        header = ""
        header_written = False
        iteration = 1

        for s in cursor:
            # Get the scan first so we can process the header if needed
            scan_data = s.data if "data" in s else s

            # Featurize the tensor
            t = featurize_tensor_dict(
                scan_data['outputs'], self.config["zone_size"],
                scan_data['zone_size'], scan_data['layer_id_to_name']
            )

            # Write the header if required
            if not header_written and self.config["write_header"]:
                logging.info("Processing header")
                header = "target,"
                # Add the passthrough fields names
                for pt in self.config["passthrough_fields"]:
                    header += f"{pt},"
                if self.config['include_output']:
                    header += "out,"

                for i in range(0, len(t[1])):
                    if self.config["use_full_zone_names"]:
                        header += f"{t[2][i]},"
                    else:
                        header += f"{t[1][i]},"
                if self.config['write_to_file']:
                    f.write(f'{header}\n')
                header_written = True

            # Process the scans
            if iteration % 100 == 0:
                logging.info(f"Iteration {iteration} of {doc_count}")

            if self.config["write_to_file"]:
                row = ','.join(map(str, t[0])) + "\n"

                # Add passthrough field data
                pt_fields = ""
                for pt in self.config["passthrough_fields"]:
                    # Split the pt field in case it's a nested field
                    # Like metadata.type
                    sp = pt.split(".")
                    if len(sp) == 1:
                        pt_fields += scan_data[pt] + ","
                    else:
                        pt_fields += scan_data[sp[0]][sp[1]] + ","
                if pt_fields != "":
                    row = pt_fields + row

                if self.config['include_output']:
                    row =\
                        scan_data['decoded_output'].replace(",", "") +\
                        ',' + row
                row = str(int(scan_data['ground_truth'])) + ',' + row
                f.write(row)
            if self.config["build_in_memory"]:
                row = t[0]

                # Add passthrough field data
                pt_fields = []
                for pt in self.config["passthrough_fields"]:
                    # Split the pt field in case it's a nested field
                    # Like metadata.type
                    sp = pt.split(".")
                    if len(sp) == 1:
                        pt_fields.append(scan_data[pt])
                    else:
                        pt_fields.append(scan_data[sp[0]][sp[1]])
                if len(pt_fields) > 0:
                    row = pt_fields + row

                if self.config['include_output']:
                    row = [scan_data['output'].replace(",", "")] + row
                row = [int(scan_data['ground_truth'])] + row
                data.append(row)
            iteration += 1

        # Build return values
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
