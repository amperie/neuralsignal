

def is_layer_string_match_in_list(lyr_name: str, layer_list: list) -> bool:
    for lyr_match in layer_list:
        if lyr_match in lyr_name:
            return True
    return False


def get_current_zone_size(lyr, current_zone_sizes):
    if lyr in current_zone_sizes:
        return current_zone_sizes[lyr]
    else:
        return current_zone_sizes['default']


def get_layer_zone_size(
        layer_name: str, zones_by_layer: dict,
        default_zone_size: int
        ) -> int:
    """
    Returns the zone size for a given layer name.
    This is the zone size the layer will be compressed to

    Args:
        layer_name (str): The name of the layer.
        zones_by_layer (dict): A dictionary mapping layer names to zone sizes.
        default_zone_size (int): The default zone size to return if no match 
        is found.

    Returns:
        int: The zone size for the given layer name. If no match is found, 
        returns the default zone size.
    """
    for lyr_match in zones_by_layer.keys():
        if lyr_match in layer_name:
            return zones_by_layer[lyr_match]
    return default_zone_size


def include_layer(
        layer_names_to_include, layer_name,
        layer_indexes_to_include, curr_idx
        ) -> bool:

    inc_layer_name = (
        is_layer_string_match_in_list(
            layer_name, layer_names_to_include
            )
        if layer_names_to_include is not None else False
        )
    inc_index_layer = (
        curr_idx in layer_indexes_to_include
        if layer_indexes_to_include is not None else False
        )
    inc_all_layers = (
        layer_indexes_to_include is None and
        layer_names_to_include is None
    )
    # If we're not including this layer, skip it
    # Skip if neither is included
    # Also skip if both params are set to None
    # which means we include all layers
    if not (inc_layer_name or inc_index_layer) and\
            not inc_all_layers:
        retVal = False
    else:
        retVal = True
    return retVal


def transform_tensor_dict_into_columns(
        tensor_dict: dict):
    pass


def transform_tensor_dict_into_pandas(
        tensor_dict: dict):
    pass
