"""
Author: Dimas Ahmad
Description: This file contains definitions of the single network and ensemble model as described in the original paper.
"""

import torch
import os


class DeepNetwork(torch.nn.Module):
    '''
    The module class performs building network according to config
    '''

    def __init__(self, config):
        super(DeepNetwork, self).__init__()
        # parses parameters of network from configuration
        self.dropout = config['dropout']
        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.input_dim = config['input_dim']

        # builds network
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            self.hidden_layers.append(torch.nn.Linear(input_dim, self.hidden_dim[i]))

        if self.dropout > 0:
            self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.output_layer = torch.nn.Linear(self.hidden_dim[-1], 1)

    def forward(self, X):
        for layer in self.hidden_layers:
            X = layer(X)
            X = torch.nn.functional.relu(X)
            if self.dropout > 0:
                X = self.dropout_layer(X)
        return self.output_layer(X).squeeze(-1)


class DeepEnsemble():
    def __init__(self, config, model_dirs):
        self._model_dirs = model_dirs
        self._model = DeepNetwork(config).to(device=config['device'])

    def predict(self, data):
        ensemble_predictions = []
        for model_file in os.listdir(self._model_dirs):
            model_path = os.path.join(self._model_dirs, model_file)
            if os.path.isfile(model_path) and model_file.endswith('.pth'):
                self._model.load_state_dict(torch.load(model_path, weights_only=False).state_dict())
                self._model.eval()
                with torch.no_grad():
                    y_pred = self._model(data)
                    ensemble_predictions.append(y_pred)
        # returns the average of the ensemble predictions
        return torch.mean(torch.stack(ensemble_predictions), dim=0)