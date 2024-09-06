import torch
from neuralsignal.core.modules.feature_sets.feature_set_base\
    import FeatureSetBase
from neuralsignal.core.modules.feature_sets.feature_utils\
    import is_layer_string_match_in_list
import pandas as pd


class FeatureSetLayerDistribution(FeatureSetBase):

    def __init__(self, config: dict):
        """
        cfg must contain the following configuration:
        layers_to_process: list of layer name string matches to process
        bin_count: how many bins to use to capture distribution
        field_to_process: name of the field to process
            (can be inputs, outputs, or deltas)
        """
        super().__init__(config)

    def get_feature_set_name(self) -> str:
        return "layer_distribution"

    def process_feature_set(self, scan: dict):
        """
        Process the zone sizes by layer for a given scan and configuration.

        Args:
            scan (dict): A dictionary containing the scan data.

        Returns: Depending on the config parameter output_format outputs are
        "name_and_value_columns", "tensor_dict" and "pandas"
        tensor_dict:
            tuple: Tensor dictionary, Dictionary of zone sizes
        name_and_value_columns:
            tuple: column names, column values
        pandas:
            DataFrame: contains the features and columns in a pandas df

        """
        self.scan = scan
        cols = []
        vals = []
        idx = 0

        layers_to_process = self.config['layers_to_process']
        field_to_process = self.config['field_to_process']
        bin_count = self.config['bin_count']

        for i, lyr in enumerate(scan['layer_order']):
            lyr_name = scan['layer_id_to_name'][lyr]

            if is_layer_string_match_in_list(lyr_name, layers_to_process):
                # Layer is in the list to process
                if field_to_process == 'deltas':
                    t = scan['outputs'][lyr] - scan['inputs'][lyr]
                else:
                    t = scan[field_to_process][lyr]
                # t = t.to(self.dev_map)
            # Build feature names for all bins
            for b in range(bin_count):
                col_name = f"bin_{b}_{field_to_process}_{lyr_name}_{idx}"
                col_name = self.make_column_name(col_name)
                cols.append(col_name)

            hist = torch.histc(t, bin_count)
            vals.append(hist.tolist())

        # Return the right format results
        output_format = self.config['output_format']
        if output_format == "name_and_value_columns":
            return (cols, vals)
        elif output_format == "tensor_dict":
            return None
        elif output_format == "pandas":
            return pd.DataFrame([vals], columns=cols)
        else:
            raise ValueError(
                "output_format must be one of 'name_and_value_columns', "
                "'tensor_dict' or 'pandas'"
                )
