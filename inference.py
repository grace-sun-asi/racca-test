"""
Inference Script — Load trained model and predict on new data
--------------------------------------------------------------
Uses the saved encoders + scaler from training to transform new data
through the same feature pipeline, then runs the model.

Usage:
    python inference.py --model etac --data new_data.csv
    python inference.py --model dccs --data new_data.csv --checkpoint trained_model_dccs.pt
"""

import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from model import FlexibleMLP
from data_loader import load_feature_config, prepare_features


def load_trained_model(checkpoint_path: str, device: torch.device = None):
    """
    Load a trained model from checkpoint.

    Returns
    -------
    model : FlexibleMLP
    checkpoint : dict (contains encoders, scaler, feature_names, etc.)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = FlexibleMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["model_config"]["hidden_dims"],
        output_dim=checkpoint["output_dim"],
        activation=checkpoint["model_config"]["activation"],
        dropout=checkpoint["model_config"]["dropout"],
        batch_norm=checkpoint["model_config"]["batch_norm"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def predict(
    df: pd.DataFrame,
    model_name: str,
    checkpoint_path: str,
    config_path: str = "feature_columns.json",
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Run inference on new data.

    Parameters
    ----------
    df : pd.DataFrame
        New data with the same columns as training data.
    model_name : str
        Model target name (etac, dccs, csdt).
    checkpoint_path : str
        Path to saved .pt checkpoint.
    config_path : str
        Path to feature_columns.json.
    device : torch.device or None

    Returns
    -------
    pd.DataFrame
        Original data with PREDICTION, PREDICTION_CONFIDENCE, and PredictionDate
        columns added, filtered to the published_columns from feature_columns.json.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and saved preprocessing artifacts
    model, checkpoint = load_trained_model(checkpoint_path, device)

    # Reconstruct scaler from saved parameters
    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]
    scaler.var_ = checkpoint["scaler_scale"] ** 2
    scaler.n_features_in_ = len(checkpoint["scaler_mean"])

    # Get saved encoders
    encoders = checkpoint["encoders"]

    # Prepare features using the SAME encoders/scaler from training
    X, _, _, _, feature_names = prepare_features(
        df=df,
        model_name=model_name,
        config_path=config_path,
        target_column="__none__",  # No target needed for inference
        fit_encoders=False,
        encoders=encoders,
        scaler=scaler,
    )

    # Run inference
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
        confidence = probabilities.max(dim=1).values.cpu().numpy()

    # Decode predictions back to original labels if a label encoder was used
    if "__label_encoder__" in encoders:
        label_enc = encoders["__label_encoder__"]
        prediction_labels = label_enc.inverse_transform(predictions)
    else:
        prediction_labels = predictions

    # Attach predictions to the DataFrame
    result_df = df.copy()
    result_df["PREDICTION"] = prediction_labels
    result_df["PREDICTION_CONFIDENCE"] = np.round(confidence, 4)
    result_df["PredictionDate"] = datetime.now().strftime("%Y-%m-%d")

    # Filter to published columns from feature_columns.json
    feature_config = load_feature_config(config_path)
    published_columns = feature_config[model_name]["published_columns"]

    # Only keep columns that exist in the result
    output_cols = [c for c in published_columns if c in result_df.columns]
    # Add confidence as a bonus column
    if "PREDICTION_CONFIDENCE" not in output_cols:
        output_cols.append("PREDICTION_CONFIDENCE")

    return result_df[output_cols]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MLP inference on new data")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["etac", "dccs", "csdt"],
        help="Model target",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to CSV data file for prediction",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt model checkpoint (default: trained_model_<model>.pt)",
    )
    parser.add_argument(
        "--config", type=str, default="feature_columns.json",
        help="Path to feature_columns.json",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save predictions CSV (default: predictions_<model>.csv)",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or f"trained_model_{args.model}.pt"
    output_path = args.output or f"predictions_{args.model}.csv"

    # Load data
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    # Run predictions
    predictions_df = predict(
        df=df,
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        config_path=args.config,
    )

    # Save results
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to '{output_path}'")
    print(f"Columns: {list(predictions_df.columns)}")
    print(f"\nSample predictions:")
    print(predictions_df.head(10).to_string())
