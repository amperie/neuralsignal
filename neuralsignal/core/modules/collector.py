import torch
import logging
from neuralsignal.core.modules.tensors import process_tensor_dict_into_zones
from neuralsignal.core.modules.tensors\
    import process_tensor_dict_into_zones_by_layer
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


class Collector:
    """Collects instrumentation data
    """

    default_config = {
        "mode": "additive",
        "data_to_save":
            ["inputs", "outputs", "layer_info", "topology"],
        "zone_size": 512,
        "zone_size_by_layer": {},
        # Entries are: "layer string to match": zone size for layer
        "layer_indexes_to_include": None,
        # Integer index of layers to include in results. none means all
        "layer_names_to_include": None,
        # String matches of layers to include in results. none means all
        "abort_on_layer_index:": 0,
        # 0 means don't abort
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
        if "zone_size_by_layer" in self.config:
            self.zone_size = self.config['zone_size_by_layer']['default']
        else:
            self.zone_size = self.config['zone_size']
        self.abort_on_layer_index = self.config['abort_on_layer_index']
        self.curr_layer_index = 0

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
        # Abort if we have reached the abort_on_layer_index
        if self.abort_on_layer_index > 0 and\
                self.curr_layer_index >= self.abort_on_layer_index:
            logging.debug(f"Aborting at layer index {self.curr_layer_index}")
            raise RuntimeError(
                f"Aborting LLM due to abort_on_layer_index "
                f"{self.curr_layer_index}")
        mod_id = str(id(module))
        dts = self.data_to_save
        batch_size = module_in[0].shape[0]
        self.batch_size = batch_size
        for batch_idx in range(batch_size):
            if "inputs" in dts:
                self.store_inputs(mod_id, module_in[0], batch_idx)
            if "outputs" in dts:
                self.store_outputs(mod_id, module_out, batch_idx)
        if "layer_info" in dts:
            self.store_layer_info(mod_id, module, module_in)
        self.curr_layer_index += 1

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
        zones_by_layer = self.config["zone_size_by_layer"]
        layer_indexes_to_include = self.config["layer_indexes_to_include"]
        layer_names_to_include = self.config["layer_names_to_include"]

        if self.mode == "additive":
            if "inputs" in self.data_to_save:
                retVal["inputs"] = {}
                for batch_idx in range(self.batch_size):
                    if not zones_by_layer:
                        # We are not doing different zone size per layer
                        rv = process_tensor_dict_into_zones(
                            self.inputs[batch_idx],
                            self.zone_size)
                        zbl = {"default": self.zone_size}
                    else:
                        # We are doing different zone size per layer
                        ret_zbl = process_tensor_dict_into_zones_by_layer(
                            self.inputs[batch_idx], zones_by_layer,
                            self.layer_id_to_name, {'default': 1},
                            self.zone_size, layer_indexes_to_include,
                            layer_names_to_include
                            )
                        retVal["layer_indexes_to_include"] =\
                            layer_indexes_to_include
                        retVal["layer_names_to_include"] =\
                            layer_names_to_include
                        retVal['zones_by_layer'] =\
                            zones_by_layer
                        rv = ret_zbl[0]
                        zbl = ret_zbl[1]
                        zbl['default'] = self.zone_size
                    retVal["inputs"][batch_idx] = rv
                    retVal["zone_sizes_by_layer"] = zbl
                self.inputs = retVal["inputs"]
                self.zone_sizes_by_layer = zbl
            if "outputs" in self.data_to_save:
                retVal["outputs"] = {}
                for batch_idx in range(self.batch_size):
                    if not zones_by_layer:
                        # We are not doing different zone size per layer
                        rv = process_tensor_dict_into_zones(
                            self.outputs[batch_idx],
                            self.zone_size)
                        zbl = {"default": self.zone_size}
                    else:
                        # We are doing different zone size per layer
                        ret_zbl = process_tensor_dict_into_zones_by_layer(
                            self.outputs[batch_idx], zones_by_layer,
                            self.layer_id_to_name, {'default': 1},
                            self.zone_size, layer_indexes_to_include,
                            layer_names_to_include
                            )
                        retVal["layer_indexes_to_include"] =\
                            layer_indexes_to_include
                        retVal["layer_names_to_include"] =\
                            layer_names_to_include
                        retVal['zones_by_layer'] =\
                            zones_by_layer
                        rv = ret_zbl[0]
                        zbl = ret_zbl[1]
                        zbl['default'] = self.zone_size
                    retVal["outputs"][batch_idx] = rv
                    retVal["zone_sizes_by_layer"] = zbl
                self.outputs = retVal["outputs"]
                self.zone_sizes_by_layer = zbl
            retVal["zone_sizes_by_layer"] = zbl
        else:
            raise NotImplementedError("Only additive mode is supported")

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
            retVal["zone_sizes_by_layer"] = self.zone_sizes_by_layer
        if "outputs" in self.data_to_save:
            retVal["outputs"] = self.outputs[batch_idx]
            retVal["zone_sizes_by_layer"] = self.zone_sizes_by_layer
        if "topology" in self.data_to_save:
            retVal["topology"] = self.topology
        if "layer_info" in self.data_to_save:
            retVal['layer_order'] = self.layer_order
            retVal['layer_names'] = self.layer_names
            retVal['layer_id_to_name'] = self.layer_id_to_name
            retVal['layer_passes'] = self.layer_passes
            retVal['layer_passes_by_name'] = self.layer_passes_by_name

        if "layer_indexes_to_include" in self.config:
            retVal['layer_indexes_to_include'] =\
                self.config["layer_indexes_to_include"]
        if "layer_names_to_include" in self.config:
            retVal['layer_names_to_include'] =\
                self.config["layer_names_to_include"]
        if "zone_size_by_layer" in self.config:
            retVal['zone_size_by_layer'] =\
                self.config["zone_size_by_layer"]

        retVal['zone_size'] = self.zone_size
        return retVal
