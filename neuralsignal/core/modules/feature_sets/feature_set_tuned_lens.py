import torch
import logging
import torch.nn as nn
import torch.optim as optim
from neuralsignal.core.modules.feature_sets.feature_set_base\
    import FeatureSetBase
from neuralsignal.core.modules.feature_sets.feature_utils\
    import is_layer_string_match_in_list
import pandas as pd
from transformers import AutoModelForSeq2SeqLM
from huggingface_hub import login
from neuralsignal.core.modules.neuralsignal_config import sdk_config

logging.basicConfig(level=sdk_config.logging_level())


class FeatureSetTunedLens(FeatureSetBase):

    def _load_model(self, model_name: str, hf_token: str):

        login(hf_token)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                                model_name,
                                device_map=self.dev_map,
                                )
        self.unembed = self.model.lm_head

    def __init__(self, config: dict):
        """
        cfg must contain the following configuration:
        model_name: model from where to get the unembedding matrix
        hf_token: huggingface token
        layers_to_process: list of layer name string matches to process
        """
        super().__init__(config)
        if torch.cuda.is_available():
            dev_map = "cuda:0"
        else:
            dev_map = "cpu"
        self.dev_map = dev_map

        if "unembed_layer" in config:
            self.unembed = config["unembed_layer"]
        else:
            self._load_model(config['model_name'], config['hf_token'])
        self.training_data = []
        self.ground_truth = []
        self.trained_model = None

    def get_feature_set_name(self) -> str:
        return "tuned-lens"

    def process_feature_set(self, scan: dict):
        """
        Process the zone sizes by layer for a given scan and configuration.

        Args:
            scan (dict): A dictionary containing the scan data.

        Returns: Depending on the config parameter output_format outputs are
        "name_and_value_columns", "tensor_dict" and "pandas"
        tensor_dict:
            tuple: Tensor dictionary, Dictionary of zone sizes
        name_and_value_columns:
            tuple: column names, column values
        pandas:
            DataFrame: contains the features and columns in a pandas df

        """
        self.scan = scan
        cols = []
        vals = []
        idx = 0

        layers_to_process = self.config['layers_to_process']
        for lyr in scan['outputs'].keys():
            lyr_name = scan['layer_id_to_name'][lyr]
            if is_layer_string_match_in_list(lyr_name, layers_to_process):
                # Layer is in the list to process
                # Get the logits from it by feeding it into the unembed matrix
                t = scan['outputs'][lyr]
                t = t.to(self.dev_map)
                logits = self.unembed.forward(t)
                cols.append(self.make_column_name(f"std_{lyr_name}_{idx}"))
                vals.append(torch.std(logits).item())
                cols.append(self.make_column_name(f"mean_{lyr_name}_{idx}"))
                vals.append(torch.mean(logits).item())
                idx += 1

        # Return the right format results
        output_format = self.config['output_format']
        if output_format == "name_and_value_columns":
            return (cols, vals)
        elif output_format == "tensor_dict":
            return None
        elif output_format == "pandas":
            return pd.DataFrame([vals], columns=cols)
        else:
            raise ValueError(
                "output_format must be one of 'name_and_value_columns', "
                "'tensor_dict' or 'pandas'"
                )

    def process_training_data(self, scan_iterator):
        """
        Process the training data for the feature set.
        """

        idx = 0
        for scan in scan_iterator:
            layers_to_process = self.config['layers_to_process']
            for lyr in scan['outputs'].keys():
                lyr_name = scan['layer_id_to_name'][lyr]
                if is_layer_string_match_in_list(lyr_name, layers_to_process):
                    # Layer is in the list to process
                    # Get the logits from it by feeding it into the unembed
                    t = scan['outputs'][lyr]
                    t = t.to(self.dev_map)
                    logits = self.unembed.forward(t[-1])
                    self.training_data.append(logits)
                    self.ground_truth.append(scan['ground_truth'])
                    idx += 1
        logging.debug(f"Training data length: {len(self.training_data)}")

    def train_feature_set(self, training_config: dict):
        """
        Train the model for the tuned lens. Config should include:
        logits_dim: dimension of the logits
        hidden_layer_dim: dimension of the hidden layer
        training_split: percentage of data to use for training
        epochs: number of epochs
        batch_size: batch size
        """
        dim = training_config['logits_dim']
        hl_dim = training_config['hidden_layer_dim']
        training_split = training_config['training_split']
        n_epochs = training_config['epochs']
        batch_size = training_config['batch_size']
        lr = training_config['learning_rate']
        dataset_size = len(self.training_data)
        split = int(dataset_size*training_split)
        X = self.training_data[0:split]
        y = self.ground_truth[0:split]
        X_val = self.training_data[split:]
        y_val = self.ground_truth[split:]

        X = torch.stack(X)
        X_val = torch.stack(X_val)

        model = nn.Sequential(
            nn.Linear(dim, hl_dim),
            nn.ReLU(),
            nn.Linear(hl_dim, int(hl_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hl_dim/2), 1),
            nn.Sigmoid(),
        ).to(self.dev_map)

        loss_fn = nn.BCELoss()  # binary cross entropy
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            for i in range(0, len(X), batch_size):
                Xbatch = X[i:i+batch_size]
                y_pred = model(Xbatch)
                ybatch = y[i:i+batch_size]
                ybatch = torch.Tensor(ybatch)[:, None]
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                logging.debug(
                    f'TunedLens training epoch {epoch}, batch {i}, loss {loss}')
            logging.debug(
                f'TunedLens training epoch {epoch}, last loss {loss}')

        self.model = model

        with torch.no_grad():
            y_pred = model(X_val)

        y_val = torch.Tensor(y_val)[:, None]
        accuracy = (y_pred.round() == y_val).float().mean()
        logging.info(f"TunedLens accuracy {accuracy}")
