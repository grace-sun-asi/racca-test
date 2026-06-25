"""
Flexible Multi-Layer Perceptron (MLP) for Classification
---------------------------------------------------------
Configurable architecture: number of layers, nodes per layer, activation,
dropout, and batch normalization are all parameterized.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class FlexibleMLP(nn.Module):
    """
    A flexible MLP classifier that supports:
    - Arbitrary number of hidden layers and nodes per layer
    - Configurable activation function
    - Optional dropout per layer
    - Optional batch normalization
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
        """
        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dims : list of int
            Number of neurons in each hidden layer. Length determines depth.
            e.g. [128, 64, 32] gives a 3-hidden-layer network.
        output_dim : int
            Number of output classes.
        activation : str
            Activation function name: 'relu', 'leaky_relu', 'tanh', 'selu', 'gelu'.
        dropout : float
            Dropout probability applied after each hidden layer (0 = no dropout).
        batch_norm : bool
            Whether to apply batch normalization after each hidden layer.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build activation
        self.activation_fn = self._get_activation(activation)

        # Build layers dynamically
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self.activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # Output layer (no activation — handled by loss function)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Weight initialization
        self._initialize_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Return an activation module by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
        }
        if name not in activations:
            raise ValueError(
                f"Unknown activation '{name}'. Choose from {list(activations.keys())}"
            )
        return activations[name]

    def _initialize_weights(self):
        """Xavier/He initialization depending on activation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (pre-softmax)."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)
