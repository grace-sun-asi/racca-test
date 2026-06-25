"""
Ensemble Model — CatBoost (Primary) + MLP (Secondary)
------------------------------------------------------
CatBoost handles categorical features natively (no one-hot encoding needed).
MLP operates on the one-hot encoded feature matrix.
Final predictions are a weighted average of both models' class probabilities.

The ensemble typically outperforms either model alone by 1-3% accuracy.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from catboost import CatBoostClassifier

from model import FlexibleMLP


class CatBoostMLP_Ensemble:
    """
    Ensemble classifier combining CatBoost (primary) and MLP (secondary).

    CatBoost:
    - Handles categorical features directly (no one-hot encoding)
    - Strong on tabular data with medium-to-high cardinality categoricals
    - Provides feature importance for interpretability

    MLP:
    - Operates on one-hot encoded + scaled features
    - Captures non-linear interactions differently than trees
    - Adds diversity to the ensemble

    Final prediction = weighted average of class probabilities:
        P(class) = catboost_weight * P_catboost + mlp_weight * P_mlp
    """

    def __init__(
        self,
        catboost_params: Dict[str, Any] = None,
        mlp_input_dim: int = None,
        mlp_hidden_dims: List[int] = None,
        mlp_output_dim: int = None,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.2,
        mlp_batch_norm: bool = True,
        catboost_weight: float = 0.6,
        mlp_weight: float = 0.4,
        device: torch.device = None,
    ):
        """
        Parameters
        ----------
        catboost_params : dict or None
            CatBoost hyperparameters. Uses sensible defaults if None.
        mlp_input_dim : int
            Number of input features for the MLP (after one-hot encoding).
        mlp_hidden_dims : list of int
            Hidden layer sizes for MLP.
        mlp_output_dim : int
            Number of output classes.
        mlp_activation : str
            Activation function for MLP.
        mlp_dropout : float
            Dropout rate for MLP.
        mlp_batch_norm : bool
            Whether to use batch norm in MLP.
        catboost_weight : float
            Weight for CatBoost predictions in ensemble (0-1).
        mlp_weight : float
            Weight for MLP predictions in ensemble (0-1).
        device : torch.device
            Device for MLP inference.
        """
        self.catboost_weight = catboost_weight
        self.mlp_weight = mlp_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- CatBoost setup ---
        self.catboost_params = catboost_params or {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 50,
            "eval_metric": "Accuracy",
            "auto_class_weights": "Balanced",  # Handles class imbalance
            "early_stopping_rounds": 30,
        }
        self.catboost_model = None

        # --- MLP setup ---
        self.mlp_input_dim = mlp_input_dim
        self.mlp_hidden_dims = mlp_hidden_dims or [128, 64, 32]
        self.mlp_output_dim = mlp_output_dim
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_model = None

        # Track training state
        self.is_fitted = False
        self.n_classes = None
        self.cat_feature_indices = None

    def _build_mlp(self):
        """Initialize the MLP model."""
        self.mlp_model = FlexibleMLP(
            input_dim=self.mlp_input_dim,
            hidden_dims=self.mlp_hidden_dims,
            output_dim=self.mlp_output_dim,
            activation=self.mlp_activation,
            dropout=self.mlp_dropout,
            batch_norm=self.mlp_batch_norm,
        )
        self.mlp_model.to(self.device)

    def _build_catboost(self):
        """Initialize the CatBoost model."""
        self.catboost_model = CatBoostClassifier(**self.catboost_params)

    def get_catboost_feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """Get feature importance from the CatBoost model."""
        if self.catboost_model is None:
            raise ValueError("CatBoost model not trained yet.")

        importances = self.catboost_model.get_feature_importance()

        if feature_names is not None:
            return dict(sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            ))
        return importances

    def predict_proba_catboost(self, X_cat: np.ndarray) -> np.ndarray:
        """Get class probabilities from CatBoost."""
        return self.catboost_model.predict_proba(X_cat)

    def predict_proba_mlp(self, X_mlp: np.ndarray) -> np.ndarray:
        """Get class probabilities from MLP."""
        self.mlp_model.eval()
        X_tensor = torch.FloatTensor(X_mlp).to(self.device)
        with torch.no_grad():
            logits = self.mlp_model(X_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict_proba(
        self,
        X_cat: np.ndarray,
        X_mlp: np.ndarray,
    ) -> np.ndarray:
        """
        Get ensemble class probabilities (weighted average).

        Parameters
        ----------
        X_cat : np.ndarray
            Features for CatBoost (can include raw categoricals as strings).
        X_mlp : np.ndarray
            Features for MLP (one-hot encoded + scaled).

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        proba_cat = self.predict_proba_catboost(X_cat)
        proba_mlp = self.predict_proba_mlp(X_mlp)

        # Weighted average
        ensemble_proba = (
            self.catboost_weight * proba_cat +
            self.mlp_weight * proba_mlp
        )
        return ensemble_proba

    def predict(
        self,
        X_cat: np.ndarray,
        X_mlp: np.ndarray,
    ) -> np.ndarray:
        """Get ensemble predicted class indices."""
        proba = self.predict_proba(X_cat, X_mlp)
        return np.argmax(proba, axis=1)

    def get_config(self) -> Dict[str, Any]:
        """Return full ensemble configuration for saving."""
        return {
            "catboost_params": self.catboost_params,
            "catboost_weight": self.catboost_weight,
            "mlp_weight": self.mlp_weight,
            "mlp_input_dim": self.mlp_input_dim,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "mlp_output_dim": self.mlp_output_dim,
            "mlp_activation": self.mlp_activation,
            "mlp_dropout": self.mlp_dropout,
            "mlp_batch_norm": self.mlp_batch_norm,
            "n_classes": self.n_classes,
            "cat_feature_indices": self.cat_feature_indices,
        }
