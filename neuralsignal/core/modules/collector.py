import torch
from neuralsignal.core.modules.tensors import process_tensor_dict_into_zones


class Collector:
    """Collects instrumentation data
    """

    default_config = {
        "mode": "additive",
        "data_to_save":
            ["inputs", "outputs", "layer_info", "topology"],
        "zone_size": 512,
    }

    def __init__(self, config: dict = None) -> None:
        if config is None:
            config = self.default_config
        self.config = {**self.default_config, **config}

        # Data structures to store the collected data
        self.inputs = []
        self.outputs = []
        self.modules = []
        self.last_layer = None

        # Configuration variables
        self.mode = self.config['mode']
        self.data_to_save = self.config['data_to_save']
        self.zone_size = self.config['zone_size']

        # Additive mode
        if self.mode == "additive":
            self.inputs = {}
            self.outputs = {}
            self.layer_order = []
            self.layer_names = []
            self.layer_id_to_name = {}
            self.layer_passes = {}
            self.layer_passes_by_name = {}
            self.in_max_global_value = 0
            self.in_min_global_value = 0
            self.out_max_global_value = 0
            self.out_min_global_value = 0

    def reset(self):
        self.__init__(config=self.config)

    def __call__(self, module, module_in, module_out) -> None:
        mod_id = str(id(module))
        dts = self.data_to_save
        batch_size = module_in[0].shape[0]
        self.batch_size = batch_size
        for batch_idx in range(batch_size):
            if "inputs" in dts:
                self.store_inputs(mod_id, module_in[0], batch_idx)
            if "outputs" in dts:
                # Pre-process tensor matrix to 1d
                # t = tensor_mean(module_out[0][batch_idx], dim=0)
                self.store_outputs(mod_id, module_out, batch_idx)
        if "layer_info" in dts:
            self.store_layer_info(mod_id, module, module_in)

    def store_inputs(self, mod_id, module_in, batch_idx):
        if self.mode == "additive":
            if batch_idx in self.inputs and mod_id in self.inputs[batch_idx]:
                self.inputs[batch_idx][mod_id] =\
                    torch.add(
                        self.inputs[batch_idx][mod_id],
                        module_in[batch_idx])
            else:
                if batch_idx not in self.inputs:
                    self.inputs[batch_idx] = {}
                self.inputs[batch_idx][mod_id] = module_in[batch_idx]
        else:
            raise NotImplementedError("Only additive mode is supported")

    def store_outputs(self, mod_id, module_out, batch_idx):
        if self.mode == "additive":
            if batch_idx in self.outputs and mod_id in self.outputs[batch_idx]:
                self.outputs[batch_idx][mod_id] =\
                    torch.add(
                        self.outputs[batch_idx][mod_id],
                        module_out[batch_idx])
            else:
                if batch_idx not in self.outputs:
                    self.outputs[batch_idx] = {}
                self.outputs[batch_idx][mod_id] = module_out[batch_idx]
        else:
            raise NotImplementedError("Only additive mode is supported")

    def store_layer_info(self, mod_id, module, module_in):
        if self.mode == "additive":
            if mod_id in self.layer_passes:
                self.layer_passes[mod_id] = \
                    self.layer_passes[mod_id] + 1
                self.layer_passes_by_name[module.ns_name] = \
                    self.layer_passes_by_name[module.ns_name] + 1
            else:
                self.layer_id_to_name[mod_id] = module.ns_name
                self.layer_order.append(mod_id)
                self.layer_names.append(module.ns_name)
                self.layer_passes[mod_id] = 1
                self.layer_passes_by_name[module.ns_name] = 1
        else:
            raise NotImplementedError("Only additive mode is supported")

    def finish_and_get_data(self) -> dict:

        retVal = {}
        retVal['zone_size'] = self.zone_size
        retVal['batch_size'] = self.batch_size
        if self.mode == "additive":
            if self.config["zone_size"] > 1:
                if "inputs" in self.data_to_save:
                    retVal["inputs"] = {}
                    for batch_idx in range(self.batch_size):
                        rv = process_tensor_dict_into_zones(
                            self.inputs[batch_idx], self.config["zone_size"])
                        retVal["inputs"][batch_idx] = rv
                    self.inputs = retVal["inputs"]
                if "outputs" in self.data_to_save:
                    retVal["outputs"] = {}
                    for batch_idx in range(self.batch_size):
                        rv = process_tensor_dict_into_zones(
                            self.outputs[batch_idx], self.config["zone_size"])
                        retVal["outputs"][batch_idx] = rv
                    self.outputs = retVal["outputs"]

        if "topology" in self.data_to_save:
            self.topology = [
                list(self.outputs[0][key].shape) for key in self.outputs[0]
                ]
            retVal["topology"] = self.topology

        if "layer_info" in self.data_to_save:
            retVal['layer_order'] = self.layer_order
            retVal['layer_names'] = self.layer_names
            retVal['layer_id_to_name'] = self.layer_id_to_name
            retVal['layer_passes'] = self.layer_passes
            retVal['layer_passes_by_name'] = self.layer_passes_by_name
        return retVal

    def get_data_by_batch_index(self, batch_idx: int) -> dict:
        retVal = {}
        if "inputs" in self.data_to_save:
            retVal["inputs"] = self.inputs[batch_idx]
        if "outputs" in self.data_to_save:
            retVal["outputs"] = self.outputs[batch_idx]
        if "topology" in self.data_to_save:
            retVal["topology"] = self.topology
        if "layer_info" in self.data_to_save:
            retVal['layer_order'] = self.layer_order
            retVal['layer_names'] = self.layer_names
            retVal['layer_id_to_name'] = self.layer_id_to_name
            retVal['layer_passes'] = self.layer_passes
            retVal['layer_passes_by_name'] = self.layer_passes_by_name
        retVal['zone_size'] = self.zone_size
        return retVal
