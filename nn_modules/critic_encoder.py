from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: Union[int, Tuple[int]],
        hidden_dim: Union[int, List[int]],
        encoding_dim: int,
        batch_norm: bool = False,
        architecture: str = "cat",
    ):
        """
        Args:
            ...
            architecture: cat, sum
        """
        super().__init__()
        self.batch_norm = batch_norm
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.architecture = architecture

        self.activation = nn.ReLU()

        def _cat_arch_flatten(x):
            design, obs = x
            design = design.expand((obs.shape[0], design.shape[0]))  # expand batch ([batch_size, design_dim])
            inputs = torch.cat((design, obs), dim=-1)
            return inputs

        def _sum_arch_flatten(x):
            design, obs = x
            design = design.expand((obs.shape[0], design.shape[0]))
            inputs = torch.stack((design, obs), dim=-1)
            return inputs

        def _cat_arch_id(x):
            return x

        def _sum_arch_id(x):
            return x.unsqueeze(-1)

        if isinstance(input_dim, int):
            self._input_dim_flat = input_dim
            if self.architecture == "cat":
                self._prepare_input = _cat_arch_id
            else:
                self._prepare_input = _sum_arch_id
        else:
            self._input_dim_flat = np.prod(input_dim)
            if self.architecture == "cat":
                self._prepare_input = _cat_arch_flatten
            else:
                self._prepare_input = _sum_arch_flatten

        if isinstance(hidden_dim, int):
            self.linear1 = nn.Linear(self._input_dim_flat, hidden_dim)
            self.middle = nn.Identity()
            self.bn1 = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
            if self.architecture == "sum":
                self.sum_layer = nn.Linear(hidden_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, encoding_dim)
        else:
            self.linear1 = nn.Linear(self._input_dim_flat, hidden_dim[0])
            self.bn1 = nn.BatchNorm1d(hidden_dim[0]) if self.batch_norm else nn.Identity()
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                        nn.BatchNorm1d(hidden_dim[i + 1]) if batch_norm else nn.Identity(),
                        self.activation,
                    )
                    for i in range(0, len(hidden_dim) - 1)
                ]
            )
            if self.architecture == "sum":
                self.sum_layer = nn.Linear(hidden_dim[-1], hidden_dim[-1])
            self.output_layer = nn.Linear(hidden_dim[-1], encoding_dim)

    def forward(self, input):
        # x shape -- should be [batch_size, flattened shape]
        x = self._prepare_input(input)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.middle(x)
        if self.architecture == "sum":
            x = x.sum(1)
            x = self.sum_layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x
