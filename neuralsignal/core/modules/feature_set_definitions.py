import logging
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.core.modules.tensors import process_tensor_dict_into_zones

logging.basicConfig(level=sdk_config.logging_level())


# Simple zone size processor that is aware of different zone sizes per layer
def process_zone_size_by_layer(
        scan: dict, cfg: dict) -> dict:

    czs = scan['zone_sizes_by_layer']  # Current zone sizes by layer
    tzs = cfg['target_zone_size']  # What zone sizes are targeted
    default_tzs = tzs['default']  # Default if a layer isn't listed in tzs
    field_to_process = cfg['field_to_process']  # inputs or outputs?

    vals = scan[field_to_process]

    for idx, lyr in enumerate(vals.keys()):
        curr_zs = czs[lyr]
        target_zs = tzs[lyr] if lyr in tzs else default_tzs
        tensor_dict = {lyr, field_to_process[lyr]}
        out = process_tensor_dict_into_zones(
            tensor_dict, target_zs, current_zone_size=curr_zs)
        )