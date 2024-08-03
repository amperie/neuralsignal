import logging
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.core.modules.tensors import process_zones_avg

logging.basicConfig(level=sdk_config.logging_level())


# Simple zone size processor that is aware of different zone sizes per layer
def process_zone_size_by_layer(
        scan: dict, cfg: dict) -> dict:

    czs = scan['zone_sizes_by_layer']
    tzs = cfg['target_zone_size']
    default_tzs = tzs['default']
