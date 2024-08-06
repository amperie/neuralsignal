import logging
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.core.modules.tensors\
    import process_tensor_dict_into_zones_by_layer

logging.basicConfig(level=sdk_config.logging_level())


# Simple zone size processor that is aware of different zone sizes per layer
def process_zone_size_by_layer(
        scan: dict, cfg: dict) -> dict:
    """
    Process the zone sizes by layer for a given scan and configuration.

    Args:
        scan (dict): A dictionary containing the scan data.
        cfg (dict): A dictionary containing the configuration data. It should
                        have the following keys:
            - 'target_zone_size' (dict): A dictionary mapping layer IDs to
                        their target zone sizes.
            - 'default' (int): The default zone size if a layer is not
                        listed in 'target_zone_size'.
            - 'layer_names_to_include' (list): A list of layer names to
                        include in the processing.
            - 'layer_indexes_to_include' (list): A list of layer indexes to
                        include in the processing.

    Returns:
        tuple: Tensor dictionary, Dictionary of zone sizes

    """

    czs = scan['zone_sizes_by_layer']  # Current zone sizes by layer
    tzs = cfg['target_zone_size']  # What zone sizes are targeted
    default_tzs = tzs['default']  # Default if a layer isn't listed in tzs
    field_to_process = cfg['field_to_process']  # inputs or outputs?
    layer_names_to_include = cfg['layer_names_to_include']
    layer_indexes_to_include = cfg['layer_indexes_to_include']

    vals = scan[field_to_process]

    processed = process_tensor_dict_into_zones_by_layer(
        vals, tzs, scan['layer_id_to_name'], czs, default_tzs,
        layer_indexes_to_include, layer_names_to_include
    )

    # Return a list of column names and a list of column values in a tuple
    return retVal
