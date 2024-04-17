import torch
from neuralsignal.core.modules.tensors import process_tensor_dict_into_zones


class Collector:
    """Collects instrumentation data
    """

    default_config = {
        "mode": "additive",
        "data_to_save":
            ["input", "output", "module", "layer_info", "topology"],
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
        if "inputs" in dts:
            self.store_inputs(mod_id, module_in)
        if "outputs" in dts:
            self.store_outputs(mod_id, module_out)
        if "layer_info" in dts:
            self.store_layer_info(mod_id, module, module_in)

    def store_inputs(self, mod_id, module_in):
        if self.mode == "additive":
            if mod_id in self.inputs:
                self.inputs[mod_id] =\
                    torch.add(self.inputs[mod_id], module_in[0])
            else:
                self.inputs[mod_id] = module_in[0]
        else:
            self.inputs.append(module_in[0])

    def store_outputs(self, mod_id, module_out):
        if self.mode == "additive":
            if mod_id in self.outputs:
                self.outputs[mod_id] =\
                    torch.add(self.outputs[mod_id], module_out[0])
            else:
                self.outputs[mod_id] = module_out[0]
        else:
            self.outputs.append(module_out[0])

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
            self.modules.append(module)

    def finish_and_get_data(self) -> dict:

        retVal = {}
        if self.mode == "additive":
            if self.config["zone_size"] > 1:
                if "inputs" in self.data_to_save:
                    self.inputs = process_tensor_dict_into_zones(
                        self.inputs, self.config["zone_size"])
                    retVal["inputs"] = self.inputs
                if "outputs" in self.data_to_save:
                    self.outputs = process_tensor_dict_into_zones(
                        self.outputs, self.config["zone_size"])
                    retVal["outputs"] = self.outputs

        if "topology" in self.data_to_save:
            self.topology = [
                list(self.outputs[key].shape) for key in self.outputs
                ]
            retVal["topology"] = self.topology

        if "layer_info" in self.data_to_save:
            retVal['layer_order'] = self.layer_order
            retVal['layer_names'] = self.layer_names
            retVal['layer_id_to_name'] = self.layer_id_to_name
            retVal['layer_passes'] = self.layer_passes
            retVal['layer_passes_by_name'] = self.layer_passes_by_name
