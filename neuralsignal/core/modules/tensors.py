import torch
import torch.nn.functional as F


def process_zones_avg(tensor_in, reduction_ratio, zone_stride=None)\
        -> torch.Tensor:
    if tensor_in.dtype != torch.float32:
        tensor_in = tensor_in.float()
    # Account for cases where the tensor can't be reduced or the reduction
    # ratio is too high. In this case we just return a tensor that reduces
    # to a dimension of 1
    reduction_ratio = min(reduction_ratio, tensor_in.shape[-1])
    if zone_stride is None:
        zone_stride = reduction_ratio
    return F.avg_pool1d(tensor_in,
                        kernel_size=reduction_ratio, stride=zone_stride)


def process_tensor_dict_into_zones(
        tensor_dict, zone_size, zone_stride=None,
        current_zone_size=1) -> dict:

    """Takes a dict of layer -> tensor and shrinks it down to zones
    If tensor is already reduced it will try to reduce it again to the
    desired zone size"""

    if zone_size == current_zone_size:
        # Nothing to do
        return tensor_dict

    if current_zone_size > 1:
        if zone_size % current_zone_size != 0:
            raise ValueError(
                "tensor_reduction_ratio must be a factor of zone_size "
                "when using reduced_outputs. "
                f"zone_size: {zone_size}, "
                f"tensor_reduction_ratio: {current_zone_size}")
    reduction_ratio = int(zone_size / current_zone_size)
    retVal = {}
    for key in tensor_dict.keys():
        retVal[key] = process_zones_avg(
            tensor_dict[key], reduction_ratio, zone_stride)
    return retVal


def process_tensor_dict_to_lists(dict_in: dict) -> dict:
    """Takes a dict of layer -> tensor and converts it to a dict of
    layer -> list"""
    retVal = {}
    for key in dict_in.keys():
        retVal[key] = dict_in[key].tolist()
    return retVal


def reduce_tensor_into_zones(
        t_in: torch.Tensor, zone_size: int,
        current_zone_size: int)\
            -> torch.Tensor:
    reduction_ratio = zone_size / current_zone_size
    return process_zones_avg(t_in, reduction_ratio)


def featurize_delta_layers(
        layers_to_featurize: list,
        inputs: dict, outputs: dict,
        layer_id_to_name: dict):
    """
    Featurizes the delta of given layers. It uses string matchin
    so it will match substrings in layer names. ie: "Attention" will
    match "Attention.k" and "Attention.q" and anything else that
    has the substring "Attention" in it.

    Args:
        layers_to_featurize (list): List of layers to be featurized.
        inputs (dict): Dictionary of input layers.
        outputs (dict): Dictionary of output layers.
        layer_id_to_name (dict): Mapping of layer IDs to layer names.

    Returns:
        Tuple: A tuple containing lists of feature names and corresponding 
        delta values.
    """
    feature_names = []
    delta_values = []
    i = 0

    for lyr in layer_id_to_name.keys():
        # if layer_id_to_name[lyr] in layers_to_featurize:
        if any(x in layer_id_to_name[lyr] for x in layers_to_featurize):
            val = torch.mean(outputs[lyr] - inputs[lyr]).item()
            name = f"delta_{i}_{layer_id_to_name[lyr]}"
            feature_names.append(name)
            delta_values.append(val)
            i += 1

    return (feature_names, delta_values)


def featurize_deltas_by_layer_name(layer_name: str, scan: dict):
    """
    Featurizes the deltas by layer name.

    Args:
        layer_name (str): The name of the layer to featurize.
        scan (dict): The scan dictionary containing layer information.

    Returns:
        Tuple: A tuple containing lists of feature names and 
        corresponding delta values.
    """
    feature_names = []
    values = []
    for lyr, i in enumerate(scan['layer_order']):
        if layer_name in scan['layer_id_to_name'][lyr]:
            # TODO: There's got to be a better way to featurize this
            val = torch.mean(scan['outputs'][lyr]).item()
            name = f"delta_{i}_{layer_name}"
            feature_names.append(name)
            values.append(val)
    return (feature_names, values)


def featurize_tensor_dict(
        tensor_dict, zone_size, current_zone_size,
        layer_id_to_name: dict = None) -> list:
    """Takes a dict of layer -> tensor and returns a tuple of
    (zone_values, zone_indexes, zone_names)
    """
    zone_indexes = []
    zone_values = []
    zone_names = []

    outputs = process_tensor_dict_into_zones(
        tensor_dict, zone_size,
        current_zone_size=current_zone_size)

    layer_index = 0
    for layer in outputs.keys():
        t = outputs[layer]
        # If it's a multi-dimensional tensor, average across all dimensions
        if len(t.shape) > 1:
            t = torch.mean(t, dim=0)
        t = t.tolist()
        zone_count = 0
        for val in t:
            zone_index = f"z_{layer_index}_{zone_count}"
            if layer_id_to_name is not None:
                layer_name = layer_id_to_name[layer]
            else:
                layer_name = "layer"
            zone_values.append(val)
            zone_indexes.append(zone_index)
            zone_names.append(f"{layer_name}_{layer_index}_{zone_count}")

            zone_count += 1
        layer_index += 1
    return (zone_values, zone_indexes, zone_names)


def tensor_mean(t_in: torch.Tensor, dim=0) -> torch.Tensor:
    return torch.mean(t_in, dim=dim)
