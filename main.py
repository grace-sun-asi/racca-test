"""
Main Script — MLP Classification with Cross-Validation
-------------------------------------------------------
Adapts to features defined in feature_columns.json.
Supports multiple model targets: etac, dccs, csdt.

Usage:
    python main.py --model etac --data path/to/data.csv
    python main.py --model dccs --data path/to/data.csv
    python main.py --model etac  (uses synthetic demo data)
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import FlexibleMLP
from train import cross_validate, train_model
from data_loader import (
    load_feature_config,
    prepare_features,
    get_model_names,
    print_feature_summary,
)


def main(
    model_name: str = "etac",
    data_path: str = None,
    config_path: str = "feature_columns.json",
    target_column: str = "PREDICTION",
):
    """
    Run the full pipeline for a specific model target.

    Parameters
    ----------
    model_name : str
        Which model to train: 'etac', 'dccs', or 'csdt'.
    data_path : str or None
        Path to CSV data file. If None, generates synthetic demo data.
    config_path : str
        Path to feature_columns.json.
    target_column : str
        Name of the target column in the dataset.
    """
    print(f"Model target: {model_name}")
    print_feature_summary(model_name, config_path)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    feature_config = load_feature_config(config_path)
    model_cfg = feature_config[model_name]
    dummy_columns = model_cfg["dummy_feature_columns"]
    additional_columns = model_cfg["additional_feature_columns"]

    if data_path is not None:
        # Load real data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows from {data_path}")
    else:
        # Generate synthetic demo data matching the feature schema
        print("No data_path provided — generating synthetic demo data...")
        df = _generate_synthetic_data(dummy_columns, additional_columns, target_column)
        print(f"Generated {len(df)} synthetic samples")

    # ------------------------------------------------------------------
    # 2. Prepare features using feature_columns.json
    # ------------------------------------------------------------------
    X, y, encoders, scaler, feature_names = prepare_features(
        df=df,
        model_name=model_name,
        config_path=config_path,
        target_column=target_column,
        fit_encoders=True,
    )

    n_classes = len(np.unique(y))
    input_dim = X.shape[1]

    print(f"\nPrepared features:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Input features (after encoding): {input_dim}")
    print(f"  Output classes: {n_classes}")
    print(f"  Class distribution: {np.bincount(y)}")

    # ------------------------------------------------------------------
    # 3. Define model architecture config
    # ------------------------------------------------------------------
    # Scale hidden layer sizes relative to input dimension
    model_config = {
        "hidden_dims": [
            min(256, input_dim * 2),   # Layer 1: expand
            min(128, input_dim),       # Layer 2: match input
            64,                        # Layer 3: compress
        ],
        "activation": "relu",
        "dropout": 0.2,
        "batch_norm": True,
    }

    # ------------------------------------------------------------------
    # 4. Define training config
    # ------------------------------------------------------------------
    train_config = {
        "epochs": 100,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "patience": 15,
    }

    # ------------------------------------------------------------------
    # 5. Run cross-validation
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Architecture: Input({input_dim}) -> {model_config['hidden_dims']} -> Output({n_classes})")
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
    # 6. Train final model on full dataset
    # ------------------------------------------------------------------
    print("\n\nTraining final model on full dataset...")

    final_model = FlexibleMLP(
        input_dim=input_dim,
        hidden_dims=model_config["hidden_dims"],
        output_dim=n_classes,
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

    # ------------------------------------------------------------------
    # 7. Save model + metadata for inference
    # ------------------------------------------------------------------
    save_path = f"trained_model_{model_name}.pt"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "model_config": model_config,
            "train_config": train_config,
            "input_dim": input_dim,
            "output_dim": n_classes,
            "model_name": model_name,
            "feature_names": feature_names,
            "encoders": encoders,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "cv_results": {
                "mean_accuracy": cv_results["mean_accuracy"],
                "std_accuracy": cv_results["std_accuracy"],
                "mean_f1": cv_results["mean_f1"],
                "std_f1": cv_results["std_f1"],
            },
        },
        save_path,
    )
    print(f"Final model saved to '{save_path}'")

    return cv_results


def _generate_synthetic_data(
    dummy_columns: list,
    additional_columns: list,
    target_column: str,
    n_samples: int = 500,
    n_classes: int = 2,
) -> pd.DataFrame:
    """
    Generate synthetic DataFrame that matches the feature_columns.json schema.
    Used for testing when real data isn't available.
    """
    np.random.seed(42)
    data = {}

    # Categorical columns — random category labels
    for col in dummy_columns:
        n_categories = np.random.randint(3, 10)
        categories = [f"{col}_cat{i}" for i in range(n_categories)]
        data[col] = np.random.choice(categories, size=n_samples)

    # Numeric/binary columns
    for col in additional_columns:
        data[col] = np.random.randint(0, 2, size=n_samples)

    # Target
    data[target_column] = np.random.randint(0, n_classes, size=n_samples)

    return pd.DataFrame(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MLP classifier")
    parser.add_argument(
        "--model", type=str, default="etac",
        choices=["etac", "dccs", "csdt"],
        help="Model target to train (default: etac)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to CSV data file (optional, uses synthetic data if not provided)",
    )
    parser.add_argument(
        "--config", type=str, default="feature_columns.json",
        help="Path to feature_columns.json",
    )
    parser.add_argument(
        "--target", type=str, default="PREDICTION",
        help="Target column name (default: PREDICTION)",
    )
    args = parser.parse_args()

    main(
        model_name=args.model,
        data_path=args.data,
        config_path=args.config,
        target_column=args.target,
    )
