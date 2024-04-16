import torch
import torch.nn.functional as F


def process_zones_avg(tensor_in, zone_size, zone_stride=None):
    if zone_stride is None:
        zone_stride = zone_size
    if tensor_in.dtype != torch.float32:
        tensor_in = tensor_in.float()
    return F.avg_pool1d(tensor_in,
                        kernel_size=zone_size, stride=zone_stride)


def process_tensor_dict_into_zones(
        tensor_dict, zone_size, zone_stride=None,
        tensor_reduction_ratio=1) -> dict:

    """Takes a dict of layer -> tensor and shrinks it down to zones
    If tensor is already reduced it will try to reduce it again to the
    desired zone size"""

    if zone_size == tensor_reduction_ratio:
        # Nothing to do
        return tensor_dict

    if tensor_reduction_ratio > 1:
        if zone_size % tensor_reduction_ratio != 0:
            raise ValueError(
                "tensor_reduction_ratio must be a factor of zone_size "
                "when using reduced_outputs. "
                f"zone_size: {zone_size}, "
                f"tensor_reduction_ratio: {tensor_reduction_ratio}")
        zone_size = int(zone_size / tensor_reduction_ratio)
    retVal = {}
    for key in tensor_dict.keys():
        retVal[key] = process_zones_avg(
            tensor_dict[key], zone_size, zone_stride)
    return retVal
