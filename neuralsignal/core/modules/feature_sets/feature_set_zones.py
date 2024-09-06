import torch
from neuralsignal.core.modules.feature_sets.feature_set_base\
    import FeatureSetBase
from neuralsignal.core.modules.tensors\
    import process_tensor_dict_into_zones_by_layer
from neuralsignal.core.modules.feature_sets.feature_utils\
    import transform_tensor_dict_into_columns
import pandas as pd


class FeatureSetZones(FeatureSetBase):

    def __init__(self, config: dict):
        super().__init__(config)

    def get_feature_set_name(self) -> str:
        return "zones"

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
        self.scan = scan
        if "zone_sizes_by_layer" in scan:
            czs = scan['zone_sizes_by_layer']  # Current zone sizes by layer
        else:
            czs = {'default': scan['zone_size']}
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
            return self._transform_tensor_dict_into_columns(processed)
        elif output_format == "tensor_dict":
            return processed
        elif output_format == "pandas":
            return self._transform_tensor_dict_into_pandas(processed)
        else:
            raise ValueError(
                "output_format must be one of 'name_and_value_columns', "
                "'tensor_dict' or 'pandas'"
                )

            return transform_tensor_dict_into_columns(processed)

    def _transform_tensor_dict_into_columns(
            self, processed: tuple
            ) -> tuple:

        td = processed[0]
        zs = processed[1]
        lyrs = self.scan['layer_id_to_name']
        col_names = []
        col_vals = []

        for included_idx, key in enumerate(td.keys()):
            t = td[key]
            if len(t.shape) > 1:
                t = torch.mean(t, dim=0)
            t = t.tolist()
            for idx, val in enumerate(t):
                zone_idx = self.scan['layer_order'].index(key)
                col_name = f"{zs[key]}_{lyrs[key]}_{included_idx}_"\
                    f"{zone_idx}_{idx}"
                col_name = self.make_column_name(col_name)

                col_names.append(col_name)
                col_vals.append(val)

        return (col_names, col_vals)

    def _transform_tensor_dict_into_pandas(
            self, processed: tuple) -> tuple:
        cols = self._transform_tensor_dict_into_columns(processed)
        return pd.DataFrame([cols[1]], columns=cols[0])
