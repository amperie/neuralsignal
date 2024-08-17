import logging
import os
import pandas as pd
from neuralsignal.backend.ns_backend import NSBackend
from neuralsignal.core.modules.tensors import featurize_tensor_dict
from neuralsignal.core.modules.tensors import featurize_delta_layers
from neuralsignal.core.modules.tensors\
    import featurize_crossmodel_deltas_by_layer_name
from neuralsignal.core.modules.tensors import featurize_layer_distributions
from neuralsignal.core.modules.tensors import featurize_embedding_vector
from neuralsignal.core.modules.feature_sets.feature_processor\
    import FeatureProcessor
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
        "featurize_delta_layers": [],  # Layer names to featurize
        "featurize_delta_by_layer_name": [],
        "featurize_zones_data": True,
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
        if "feature_processor" in config:
            self.feature_processor = config["feature_processor"]
        self.be = NSBackend(config)

    def set_feature_processor(self, feature_processor: FeatureProcessor):
        self.feature_processor = feature_processor

    def add_columns(
            self,
            col_headers: list, col_values: list,
            existing_columns: tuple
            ):
        head = existing_columns[0] + col_headers
        vals = existing_columns[1] + col_values
        return (head, vals)

    def get_column_strings(self, existing_columns: tuple):
        header = ",".join(existing_columns[0]) + "\n"
        values = ",".join(str(elem) for elem in existing_columns[1]) + "\n"
        return (header, values)

    def create_dataset_old(self, query: dict):
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

        header_written = False
        iteration = 1
        in_memory_data = []

        for s in cursor:

            if iteration % 100 == 0:
                logging.info(f"Iteration {iteration} of {doc_count}")

            # Get the scan first so we can process the header if needed
            scan_data = s.data if "data" in s else s

            # Featurize what we need

            # Featurize the tensor
            t = featurize_tensor_dict(
                scan_data['outputs'], self.config["zone_size"],
                scan_data['zone_size'], scan_data['layer_id_to_name']
            )

            # Featurize the delta layers if needed and the data is available
            if 'inputs' in scan_data and 'outputs' in scan_data:
                dlt = featurize_delta_layers(
                    self.config["featurize_delta_layers"],
                    scan_data['inputs'], scan_data['outputs'],
                    scan_data['layer_id_to_name']
                )
            else:
                logging.error(
                    "Featurizing layer deltas - no inputs/outpus fields")
                raise ValueError("No inputs/outputs in scan data to featurize")

            # Build the header and row
            header = []
            values = []
            curr_row = (header, values)

            # Add Target
            curr_row = self.add_columns(
                ['target'], [int(scan_data['ground_truth'])], curr_row)

            # Add pass-through fields
            for pt in self.config["passthrough_fields"]:
                nested_name = pt.split(".")
                if len(nested_name) == 1:
                    pt_val = str(scan_data[pt]).replace(",", "")
                else:
                    pt_val =\
                        str(scan_data[nested_name[0]][nested_name[1]])\
                        .replace(",", "")

                curr_row = self.add_columns(
                    [f"{pt}"], [f"{pt_val}"], curr_row
                )

            # Add the delta layers features
            # If there are no delta layers, dlt 0 and 1 are empty
            curr_row = self.add_columns(
                dlt[0], dlt[1], curr_row
            )

            # Add the cross model layer deltas
            for lyr in self.config["featurize_delta_by_layer_name"]:
                f_deltas = featurize_crossmodel_deltas_by_layer_name(
                    lyr, scan_data)
                curr_row = self.add_columns(
                    f_deltas[0], f_deltas[1], curr_row
                )

            # Add the layer distributions
            for lyr in self.config["featurize_layer_distributions_layers"]:
                bin_size =\
                    self.config["featurize_layer_distributions_bin_count"]
                if self.config["featurize_layer_distributions_absolute"]:
                    f_dists = featurize_layer_distributions(
                        lyr, bin_size, scan_data, False)
                    curr_row = self.add_columns(
                        f_dists[0], f_dists[1], curr_row
                    )
                if self.config["featurize_layer_distributions_deltas"]:
                    f_dists = featurize_layer_distributions(
                        lyr, bin_size, scan_data, True)
                    curr_row = self.add_columns(
                        f_dists[0], f_dists[1], curr_row
                    )

            # Add the zones data
            if self.config["featurize_zones_data"]:
                if self.config["use_full_zone_names"]:
                    curr_row = self.add_columns(
                        t[2], t[0], curr_row
                    )
                else:
                    curr_row = self.add_columns(
                        t[1], t[0], curr_row
                    )

            # Add attention layer vectors
            for lyr in self.config["featurize_embedding_vector_layers"]:
                features = featurize_embedding_vector(
                    self.config["featurize_embedding_vector_range"],
                    lyr, scan_data,
                    mode=self.config["featurize_embedding_vector_mode"]
                    )
                curr_row = self.add_columns(
                    features[0], features[1], curr_row
                )

            # Write to file and/or memory
            row_strings = self.get_column_strings(curr_row)

            # Write the header if required
            if not header_written and self.config["write_header"]\
                    and self.config['write_to_file']:
                logging.info("Processing header")
                f.write(f'{row_strings[0]}')
                header_written = True

            # Write the data
            if self.config['write_to_file']:
                f.write(f'{row_strings[1]}')

            # Now do the memory writing
            if self.config['build_in_memory']:
                in_memory_data.append(curr_row[1])
                column_names = curr_row[0]

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
                pd.DataFrame(in_memory_data, columns=column_names)

        logging.info("Done processing")
        return (file_retVal, memory_retVal)

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

        header_written = False
        iteration = 1
        in_memory_data = []
        fp = self.feature_processor

        for s in cursor:

            if iteration % 100 == 0:
                logging.info(f"Iteration {iteration} of {doc_count}")

            # Get the scan first so we can process the header if needed
            scan_data = s.data if "data" in s else s
            fp.set_scan(scan_data)

            # Build the header and row
            header = []
            values = []
            curr_row = (header, values)

            # Add Target
            curr_row = self.add_columns(
                ['target'], [int(scan_data['ground_truth'])], curr_row)

            # Add pass-through fields
            for pt in self.config["passthrough_fields"]:
                nested_name = pt.split(".")
                if len(nested_name) == 1:
                    pt_val = str(scan_data[pt]).replace(",", "")
                else:
                    pt_val =\
                        str(scan_data[nested_name[0]][nested_name[1]])\
                        .replace(",", "")

                curr_row = self.add_columns(
                    [f"{pt}"], [f"{pt_val}"], curr_row
                )

            # Featurize what we need
            features = fp.featurize()

            # Add features
            curr_row = self.add_columns(
                features[0], features[1], curr_row
            )

            # Write to file and/or memory
            row_strings = self.get_column_strings(curr_row)

            # Write the header if required
            if not header_written and self.config["write_header"]\
                    and self.config['write_to_file']:
                logging.info("Processing header")
                f.write(f'{row_strings[0]}')
                header_written = True

            # Write the data
            if self.config['write_to_file']:
                f.write(f'{row_strings[1]}')

            # Now do the memory writing
            if self.config['build_in_memory']:
                in_memory_data.append(curr_row[1])
                column_names = curr_row[0]

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
                pd.DataFrame(in_memory_data, columns=column_names)

        logging.info("Done processing")
        return (file_retVal, memory_retVal)
