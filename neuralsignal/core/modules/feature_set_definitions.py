import logging
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.core.modules.tensors\
    import process_tensor_dict_into_zones_by_layer

logging.basicConfig(level=sdk_config.logging_level())


# Simple zone size processor that is aware of different zone sizes per layer
def process_zone_size_by_layer(
        scan: dict, cfg: dict) -> dict:

    czs = scan['zone_sizes_by_layer']  # Current zone sizes by layer
    tzs = cfg['target_zone_size']  # What zone sizes are targeted
    default_tzs = tzs['default']  # Default if a layer isn't listed in tzs
    field_to_process = cfg['field_to_process']  # inputs or outputs?
    layer_names_to_include = cfg['layer_names_to_include']
    layer_indexes_to_include = cfg['layer_indexes_to_include']

    vals = scan[field_to_process]

    retVal = process_tensor_dict_into_zones_by_layer(
        vals, tzs, scan['layer_id_to_name'], czs, default_tzs,
        layer_indexes_to_include, layer_names_to_include
    )

    return retVal
