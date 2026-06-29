"""
Flexible Multi-Layer Perceptron (MLP) for Classification.
"""
import torch
import torch.nn as nn
from typing import List


class FlexibleMLP(nn.Module):
    """
    MLP classifier with configurable layers, activation, dropout, and batch norm.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()

        act_map = {
            "relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.1),
            "tanh": nn.Tanh(), "selu": nn.SELU(),
            "gelu": nn.GELU(), "elu": nn.ELU(),
        }
        if activation not in act_map:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(act_map.keys())}")
        act_fn = act_map[activation]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
