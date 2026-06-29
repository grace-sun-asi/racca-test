"""
Inference — Load saved ensemble and predict on new data.

Usage:
    python inference.py --model dccs --data new_data.csv
    python inference.py --model dccs --data new_data.csv --checkpoint ensemble_model_dccs.pt
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

from model import FlexibleMLP
from data_loader import load_feature_config, load_from_csv, load_from_sql, prepare_data


def load_ensemble(checkpoint_path: str, device: torch.device = None):
    """
    Load a saved ensemble (CatBoost + MLP) from checkpoint.

    Returns dict with: catboost, mlp, config (weights, encoders, scaler, etc.)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Rebuild MLP
    mlp = FlexibleMLP(
        input_dim=ckpt["mlp_input_dim"],
        hidden_dims=ckpt["mlp_hidden_dims"],
        output_dim=ckpt["n_classes"],
        activation="relu", dropout=0.2, batch_norm=True,
    )
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    mlp.to(device).eval()

    # Rebuild CatBoost
    cb = CatBoostClassifier()
    cb.load_model(ckpt["catboost_path"])

    # Rebuild scaler
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = ckpt["scaler_scale"] ** 2
    scaler.n_features_in_ = len(ckpt["scaler_mean"])

    return {
        "catboost": cb,
        "mlp": mlp,
        "device": device,
        "encoders": ckpt["encoders"],
        "scaler": scaler,
        "catboost_weight": ckpt["catboost_weight"],
        "model_name": ckpt["model_name"],
        "n_classes": ckpt["n_classes"],
    }


def predict(
    df: pd.DataFrame,
    model_name: str,
    checkpoint_path: str,
    config_path: str = "feature_columns.json",
) -> pd.DataFrame:
    """
    Run ensemble inference on new data.

    Returns DataFrame with PREDICTION, PREDICTION_CONFIDENCE, PredictionDate added,
    filtered to published_columns from feature_columns.json.
    """
    ensemble = load_ensemble(checkpoint_path)
    device = ensemble["device"]
    catboost_weight = ensemble["catboost_weight"]
    mlp_weight = 1.0 - catboost_weight

    # Prepare features using saved encoders/scaler
    data = prepare_data(
        df, model_name, target_column="__none__",
        config_path=config_path, fit=False,
        encoders=ensemble["encoders"], scaler=ensemble["scaler"],
    )

    X_cat, X_mlp = data["X_cat"], data["X_mlp"]

    # CatBoost predictions
    proba_cat = ensemble["catboost"].predict_proba(X_cat)

    # MLP predictions
    with torch.no_grad():
        logits = ensemble["mlp"](torch.FloatTensor(X_mlp).to(device))
        proba_mlp = torch.softmax(logits, dim=1).cpu().numpy()

    # Ensemble
    proba = catboost_weight * proba_cat + mlp_weight * proba_mlp
    predictions = np.argmax(proba, axis=1)
    confidence = proba.max(axis=1)

    # Decode labels
    encoders = ensemble["encoders"]
    if "__label_encoder__" in encoders:
        prediction_labels = encoders["__label_encoder__"].inverse_transform(predictions)
    else:
        prediction_labels = predictions

    # Build result
    result = df.copy()
    result["PREDICTION"] = prediction_labels
    result["PREDICTION_CONFIDENCE"] = np.round(confidence, 4)
    result["PredictionDate"] = datetime.now().strftime("%Y-%m-%d")

    # Filter to published columns
    config = load_feature_config(config_path)
    published = config[model_name].get("published_columns", [])
    output_cols = [c for c in published if c in result.columns]
    if "PREDICTION_CONFIDENCE" not in output_cols:
        output_cols.append("PREDICTION_CONFIDENCE")

    return result[output_cols]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble Inference")
    parser.add_argument("--model", type=str, required=True, choices=["etac", "dccs", "csdt"])
    parser.add_argument("--data", type=str, default=None, help="CSV file for prediction")
    parser.add_argument("--sql", type=str, default=None, help="SQL query for prediction data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--config", type=str, default="feature_columns.json")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    checkpoint = args.checkpoint or f"ensemble_model_{args.model}.pt"
    output_path = args.output or f"predictions_{args.model}.csv"

    # Load data
    if args.sql:
        df = load_from_sql(args.sql, args.model)
    elif args.data:
        df = load_from_csv(args.data)
    else:
        raise ValueError("Provide --data or --sql")

    # Predict
    results = predict(df, args.model, checkpoint, args.config)
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    print(f"  Rows: {len(results)}")
    print(f"  Predictions: {results['PREDICTION'].value_counts().to_dict()}")
