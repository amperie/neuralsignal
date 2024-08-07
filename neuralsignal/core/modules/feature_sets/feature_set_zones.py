from feature_set_base import FeatureSetBase
from neuralsignal.core.modules.tensors\
    import process_tensor_dict_into_zones_by_layer
from neuralsignal.core.modules.feature_sets.feature_utils\
    import transform_tensor_dict_into_columns
from neuralsignal.core.modules.feature_sets.feature_utils\
    import transform_tensor_dict_into_pandas


class FeatureSetZones(FeatureSetBase):

    def __init__(self, processing_function, config: dict):
        super().__init__(processing_function, config)

    def process_feature_set(self, scan: dict):
        """
        Process the zone sizes by layer for a given scan and configuration.

        Args:
            scan (dict): A dictionary containing the scan data.
            cfg (dict): A dictionary containing the configuration data. It
                            should have the following keys:
                - 'target_zone_size' (dict): A dictionary mapping layer IDs to
                            their target zone sizes.
                - 'default' (int): The default zone size if a layer is not
                            listed in 'target_zone_size'.
                - 'layer_names_to_include' (list): A list of layer names to
                            include in the processing.
                - 'layer_indexes_to_include' (list): A list of layer indexes to
                            include in the processing.

        Returns: Depending on the config parameter output_format outputs are
        "name_and_value_columns", "tensor_dict" and "pandas"
            tuple: Tensor dictionary, Dictionary of zone sizes

        """

        czs = scan['zone_sizes_by_layer']  # Current zone sizes by layer
        tzs = self.config['target_zone_size']  # What zone sizes are targeted
        default_tzs = tzs['default']  # Default if a layer isn't listed in tzs
        field_to_process = self.config['field_to_process']  # inputs or output?
        layer_names_to_include = self.config['layer_names_to_include']
        layer_indexes_to_include = self.config['layer_indexes_to_include']
        output_format = self.config['output_format']

        vals = scan[field_to_process]

        processed = process_tensor_dict_into_zones_by_layer(
            vals, tzs, scan['layer_id_to_name'], czs, default_tzs,
            layer_indexes_to_include, layer_names_to_include
        )

        # Return the right format results
        if output_format == "name_and_value_columns":
            return transform_tensor_dict_into_columns(processed)
        elif output_format == "tensor_dict":
            return processed
        elif output_format == "pandas":
            return transform_tensor_dict_into_pandas(processed)
        else:
            raise ValueError(
                "output_format must be one of 'name_and_value_columns', "
                "'tensor_dict' or 'pandas'"
                )
