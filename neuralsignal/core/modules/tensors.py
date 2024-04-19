import torch
import torch.nn.functional as F


def process_zones_avg(tensor_in, reduction_ratio, zone_stride=None)\
        -> torch.Tensor:
    if zone_stride is None:
        zone_stride = reduction_ratio
    if tensor_in.dtype != torch.float32:
        tensor_in = tensor_in.float()
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
        zone_size = int(zone_size / current_zone_size)
    retVal = {}
    for key in tensor_dict.keys():
        retVal[key] = process_zones_avg(
            tensor_dict[key], zone_size, zone_stride)
    return retVal


def reduce_tensor_into_zones(
        t_in: torch.Tensor, zone_size: int,
        current_zone_size: int)\
            -> torch.Tensor:
    reduction_ratio = zone_size / current_zone_size
    return process_zones_avg(t_in, reduction_ratio)


def featurize_tensor_dict(
        tensor_dict, zone_size, current_zone_size,
        layer_id_to_name: dict = None) -> list:
    """Takes a dict of layer -> tensor and returns a list of zones
    If layer_id_to_name dictionary object is there, it will return
    the names of the layers and zone index for each zone as well
    """
    zone_indexes = []
    zone_values = []
    zone_names = []

    outputs = process_tensor_dict_into_zones(
        tensor_dict, zone_size, current_zone_size)

    layer_index = 0
    for layer in outputs.keys():
        t = outputs[layer][0]
        # If it's a multi-dimensional tensor, average across all dimensions
        if len(t.shape) > 1:
            t = torch.mean(t, dim=0)
        t = t.tolist()
        zone_index = 0
        for val in t:
            zone_index = f"z{layer_index}_{zone_index}"
            if layer_id_to_name is not None:
                layer_name = layer_id_to_name[layer]
            else:
                layer_name = "layer"
            zone_values.append(val)
            zone_indexes.append(zone_index)
            zone_names.append(f"{layer_name}_{zone_index}")

            zone_index += 1
        layer_index += 1
    return (zone_values, zone_indexes, zone_names)
