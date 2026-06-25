"""
Main Script — MLP Classification with Cross-Validation
-------------------------------------------------------
Demonstrates usage of FlexibleMLP + cross_validate on a synthetic dataset.
Swap out the data loading section with your own dataset.
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from model import FlexibleMLP
from train import cross_validate, train_model
from torch.utils.data import DataLoader, TensorDataset


def main():
    # ------------------------------------------------------------------
    # 1. Generate or load data
    # ------------------------------------------------------------------
    # Synthetic dataset for demonstration — replace with your own data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=4,
        n_clusters_per_class=2,
        random_state=42,
    )

    # Standardize features (important for neural networks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Class distribution: {np.bincount(y)}")

    # ------------------------------------------------------------------
    # 2. Define model architecture config
    # ------------------------------------------------------------------
    # Base 3-layer MLP architecture
    model_config = {
        "hidden_dims": [128, 64, 32],   # 3 hidden layers
        "activation": "relu",
        "dropout": 0.2,
        "batch_norm": True,
    }

    # ------------------------------------------------------------------
    # 3. Define training config
    # ------------------------------------------------------------------
    train_config = {
        "epochs": 100,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "patience": 15,  # early stopping
    }

    # ------------------------------------------------------------------
    # 4. Run cross-validation
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Architecture: Input({X.shape[1]}) -> {model_config['hidden_dims']} -> Output({len(np.unique(y))})")
    print(f"Training: {train_config['optimizer']}, lr={train_config['lr']}, epochs={train_config['epochs']}")
    print()

    cv_results = cross_validate(
        X=X,
        y=y,
        model_config=model_config,
        train_config=train_config,
        n_folds=5,
        batch_size=32,
        device=device,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 5. Train final model on full dataset (optional)
    # ------------------------------------------------------------------
    print("\n\nTraining final model on full dataset...")

    input_dim = X.shape[1]
    output_dim = len(np.unique(y))

    final_model = FlexibleMLP(
        input_dim=input_dim,
        hidden_dims=model_config["hidden_dims"],
        output_dim=output_dim,
        activation=model_config["activation"],
        dropout=model_config["dropout"],
        batch_norm=model_config["batch_norm"],
    )

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    full_dataset = TensorDataset(X_tensor, y_tensor)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

    results = train_model(
        model=final_model,
        train_loader=full_loader,
        val_loader=None,
        config=train_config,
        device=device,
    )

    # Save the final model
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "model_config": model_config,
            "train_config": train_config,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "cv_results": {
                "mean_accuracy": cv_results["mean_accuracy"],
                "std_accuracy": cv_results["std_accuracy"],
                "mean_f1": cv_results["mean_f1"],
                "std_f1": cv_results["std_f1"],
            },
        },
        "trained_model.pt",
    )
    print("Final model saved to 'trained_model.pt'")


if __name__ == "__main__":
    main()
