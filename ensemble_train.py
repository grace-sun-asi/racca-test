"""
Ensemble Training — CatBoost (Primary) + MLP (Secondary)
---------------------------------------------------------
Trains both models, runs cross-validation, and provides inference.
CatBoost gets raw categorical data. MLP gets one-hot encoded + scaled data.

Usage:
    python ensemble_train.py --model dccs --data data.csv --target MM_Determination_Completion_Status
    python ensemble_train.py --model etac --sql "SELECT * FROM vw_etac_training"
    python ensemble_train.py --model dccs  (synthetic demo)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from model import FlexibleMLP
from ensemble_model import CatBoostMLP_Ensemble
from data_loader import (
    load_feature_config, load_from_sql, load_from_csv,
    build_derived_columns, build_binary_feature_matrix,
    add_feature_matrix_columns, print_feature_summary,
    MAX_CATEGORIES_PER_COLUMN, MAX_SAMPLE_ROWS,
)

OUTPUT_DIR = Path("output/ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# =============================================================================
# DATA PREP (dual format: raw for CatBoost, encoded for MLP)
# =============================================================================

def prepare_ensemble_data(
    df, model_name, config_path="feature_columns.json",
    target_column="PREDICTION", fit=True, encoders=None, scaler=None,
):
    """
    Prepare data in two formats:
    - X_cat: raw categoricals + numerics for CatBoost
    - X_mlp: one-hot encoded + scaled for MLP

    Returns: X_cat, X_mlp, y, cat_indices, encoders, scaler, mlp_names, cat_col_names
    """
    config = load_feature_config(config_path)
    model_cfg = config[model_name]
    dummy_columns = model_cfg["dummy_feature_columns"]
    additional_columns = model_cfg["additional_feature_columns"]

    # Build derived columns (Month_Due, Routed_Late, etc.)
    df = build_derived_columns(df.copy(), data_source=model_name)

    # --- CatBoost features (raw, no encoding needed) ---
    cat_cols = [c for c in dummy_columns if c in df.columns]
    num_cols = [c for c in additional_columns if c in df.columns]
    all_cat_feature_names = cat_cols + num_cols

    X_cat_df = df[all_cat_feature_names].copy()
    for col in cat_cols:
        X_cat_df[col] = X_cat_df[col].astype(str).fillna("Unknown")
    for col in num_cols:
        X_cat_df[col] = pd.to_numeric(X_cat_df[col], errors="coerce").fillna(0)

    cat_feature_indices = list(range(len(cat_cols)))  # indices of categorical cols
    X_cat = X_cat_df.values

    # --- MLP features (one-hot encoded + scaled) ---
    if fit:
        encoders = {}
        X_binary = build_binary_feature_matrix(df, dummy_columns)
        encoders["__feature_columns__"] = X_binary.columns.tolist()
    else:
        X_binary = build_binary_feature_matrix(df, dummy_columns)
        X_binary = X_binary.reindex(columns=encoders["__feature_columns__"], fill_value=0)

    X_mlp_df = add_feature_matrix_columns(X_binary, df, additional_columns)
    mlp_feature_names = X_mlp_df.columns.tolist()
    if fit:
        encoders["__mlp_feature_columns__"] = mlp_feature_names

    X_mlp = X_mlp_df.values.astype(np.float32)
    if fit:
        scaler = StandardScaler()
        X_mlp = scaler.fit_transform(X_mlp)
    else:
        all_cols = encoders.get("__mlp_feature_columns__", mlp_feature_names)
        X_mlp_df = X_mlp_df.reindex(columns=all_cols, fill_value=0)
        X_mlp = X_mlp_df.values.astype(np.float32)
        X_mlp = scaler.transform(X_mlp)
        mlp_feature_names = all_cols

    # --- Target ---
    if target_column in df.columns:
        y_raw = df[target_column].values
        if y_raw.dtype == object or (len(y_raw) > 0 and isinstance(y_raw[0], str)):
            if fit:
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
                encoders["__label_encoder__"] = le
            else:
                y = encoders["__label_encoder__"].transform(y_raw)
        else:
            y = y_raw.astype(np.int64)
    else:
        y = np.zeros(len(df), dtype=np.int64)

    return X_cat, X_mlp, y, cat_feature_indices, encoders, scaler, mlp_feature_names, all_cat_feature_names



# =============================================================================
# MLP TRAINING HELPER
# =============================================================================

def train_mlp(model, X_train, y_train, X_val=None, y_val=None,
              epochs=100, lr=1e-3, batch_size=32, patience=15, device=None):
    """Train MLP with optional early stopping on validation set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None:
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb), yb).item()
                    n += 1
            val_loss /= max(n, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model



# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate_ensemble(
    df, model_name, config_path="feature_columns.json",
    target_column="PREDICTION", n_folds=5,
    catboost_weight=0.6, mlp_weight=0.4,
    catboost_params=None, mlp_epochs=100, verbose=True,
):
    """Stratified K-Fold CV comparing CatBoost, MLP, and Ensemble."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_cat, X_mlp, y, cat_indices, encoders, scaler, mlp_names, cat_names = \
        prepare_ensemble_data(df, model_name, config_path, target_column, fit=True)

    n_classes = len(np.unique(y))
    mlp_input_dim = X_mlp.shape[1]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results_cat, results_mlp, results_ens, results_f1 = [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_mlp, y)):
        if verbose:
            print(f"\n--- Fold {fold+1}/{n_folds} ---")

        X_cat_tr, X_cat_va = X_cat[tr_idx], X_cat[va_idx]
        X_mlp_tr, X_mlp_va = X_mlp[tr_idx], X_mlp[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # --- CatBoost ---
        cb_params = catboost_params or {
            "iterations": 1500, "learning_rate": 0.03, "depth": 7,
            "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": 0,
            "eval_metric": "Accuracy", "auto_class_weights": "Balanced",
            "early_stopping_rounds": 50,
        }
        cb = CatBoostClassifier(**cb_params)
        cb.fit(
            Pool(X_cat_tr, y_tr, cat_features=cat_indices),
            eval_set=Pool(X_cat_va, y_va, cat_features=cat_indices),
            verbose=0,
        )
        proba_cat = cb.predict_proba(X_cat_va)
        acc_cat = accuracy_score(y_va, np.argmax(proba_cat, axis=1))

        # --- MLP ---
        mlp = FlexibleMLP(
            input_dim=mlp_input_dim,
            hidden_dims=[min(256, mlp_input_dim*2), min(128, mlp_input_dim), 64],
            output_dim=n_classes, activation="relu", dropout=0.2, batch_norm=True,
        )
        mlp = train_mlp(mlp, X_mlp_tr, y_tr, X_mlp_va, y_va, epochs=mlp_epochs, device=device)
        mlp.eval()
        with torch.no_grad():
            proba_mlp = torch.softmax(
                mlp(torch.FloatTensor(X_mlp_va).to(device)), dim=1
            ).cpu().numpy()
        acc_mlp = accuracy_score(y_va, np.argmax(proba_mlp, axis=1))

        # --- Ensemble (weighted average) ---
        proba_ens = catboost_weight * proba_cat + mlp_weight * proba_mlp
        pred_ens = np.argmax(proba_ens, axis=1)
        acc_ens = accuracy_score(y_va, pred_ens)
        f1_ens = f1_score(y_va, pred_ens, average="weighted")

        results_cat.append(acc_cat)
        results_mlp.append(acc_mlp)
        results_ens.append(acc_ens)
        results_f1.append(f1_ens)

        if verbose:
            print(f"  CatBoost: {acc_cat:.4f} | MLP: {acc_mlp:.4f} | Ensemble: {acc_ens:.4f}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"CV RESULTS ({n_folds} folds)")
        print(f"  CatBoost:  {np.mean(results_cat):.4f} +/- {np.std(results_cat):.4f}")
        print(f"  MLP:       {np.mean(results_mlp):.4f} +/- {np.std(results_mlp):.4f}")
        print(f"  Ensemble:  {np.mean(results_ens):.4f} +/- {np.std(results_ens):.4f}")
        print(f"  F1 (ens):  {np.mean(results_f1):.4f}")
        print(f"{'='*60}")

    return {
        "mean_catboost": np.mean(results_cat),
        "mean_mlp": np.mean(results_mlp),
        "mean_ensemble": np.mean(results_ens),
        "std_ensemble": np.std(results_ens),
        "mean_f1": np.mean(results_f1),
        "fold_acc_catboost": results_cat,
        "fold_acc_mlp": results_mlp,
        "fold_acc_ensemble": results_ens,
    }



# =============================================================================
# TRAIN FINAL ENSEMBLE + SAVE
# =============================================================================

def train_and_save_ensemble(
    df, model_name, config_path="feature_columns.json",
    target_column="PREDICTION", catboost_weight=0.6, mlp_weight=0.4,
    catboost_params=None, mlp_epochs=100, save_path=None, verbose=True,
):
    """Train CatBoost + MLP on full dataset and save both models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_cat, X_mlp, y, cat_indices, encoders, scaler, mlp_names, cat_names = \
        prepare_ensemble_data(df, model_name, config_path, target_column, fit=True)

    n_classes = len(np.unique(y))
    mlp_input_dim = X_mlp.shape[1]

    if verbose:
        print(f"Training ensemble: {len(y)} samples, {n_classes} classes")
        print(f"  CatBoost input: {X_cat.shape[1]} features (raw)")
        print(f"  MLP input:      {X_mlp.shape[1]} features (encoded)")

    # --- Train CatBoost (primary) ---
    cb_params = catboost_params or {
        "iterations": 800, "learning_rate": 0.03, "depth": 6,
        "l2_leaf_reg": 3.0, "random_seed": 42,
        "verbose": 50 if verbose else 0,
        "eval_metric": "Accuracy", "auto_class_weights": "Balanced",
    }
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(Pool(X_cat, y, cat_features=cat_indices))

    # --- Train MLP (secondary) ---
    mlp_model = FlexibleMLP(
        input_dim=mlp_input_dim,
        hidden_dims=[min(256, mlp_input_dim*2), min(128, mlp_input_dim), 64],
        output_dim=n_classes, activation="relu", dropout=0.2, batch_norm=True,
    )
    mlp_model = train_mlp(mlp_model, X_mlp, y, epochs=mlp_epochs, device=device)

    # --- Build ensemble object ---
    ensemble = CatBoostMLP_Ensemble(
        catboost_params=cb_params,
        mlp_input_dim=mlp_input_dim,
        mlp_hidden_dims=[min(256, mlp_input_dim*2), min(128, mlp_input_dim), 64],
        mlp_output_dim=n_classes,
        catboost_weight=catboost_weight, mlp_weight=mlp_weight, device=device,
    )
    ensemble.catboost_model = cb_model
    ensemble.mlp_model = mlp_model
    ensemble.n_classes = n_classes
    ensemble.cat_feature_indices = cat_indices
    ensemble.is_fitted = True

    # --- Save ---
    if save_path is None:
        save_path = f"ensemble_model_{model_name}.pt"

    cb_path = str(OUTPUT_DIR / f"catboost_{model_name}.cbm")
    cb_model.save_model(cb_path)

    torch.save({
        "mlp_state_dict": mlp_model.state_dict(),
        "ensemble_config": ensemble.get_config(),
        "catboost_path": cb_path,
        "encoders": encoders,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "mlp_feature_names": mlp_names,
        "cat_feature_names": cat_names,
        "model_name": model_name,
        "n_classes": n_classes,
    }, save_path)

    if verbose:
        print(f"\nSaved: CatBoost -> {cb_path}")
        print(f"Saved: Ensemble -> {save_path}")

    return ensemble, encoders


# =============================================================================
# LOAD & PREDICT (INFERENCE)
# =============================================================================

def load_ensemble(checkpoint_path, device=None):
    """Load a saved ensemble from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["ensemble_config"]

    # Rebuild MLP
    mlp = FlexibleMLP(
        input_dim=cfg["mlp_input_dim"], hidden_dims=cfg["mlp_hidden_dims"],
        output_dim=cfg["mlp_output_dim"], activation=cfg["mlp_activation"],
        dropout=cfg["mlp_dropout"], batch_norm=cfg["mlp_batch_norm"],
    )
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    mlp.to(device).eval()

    # Rebuild CatBoost
    cb = CatBoostClassifier()
    cb.load_model(ckpt["catboost_path"])

    # Rebuild ensemble wrapper
    ensemble = CatBoostMLP_Ensemble(
        catboost_params=cfg["catboost_params"],
        mlp_input_dim=cfg["mlp_input_dim"],
        mlp_hidden_dims=cfg["mlp_hidden_dims"],
        mlp_output_dim=cfg["mlp_output_dim"],
        catboost_weight=cfg["catboost_weight"],
        mlp_weight=cfg["mlp_weight"],
        device=device,
    )
    ensemble.catboost_model = cb
    ensemble.mlp_model = mlp
    ensemble.n_classes = cfg["n_classes"]
    ensemble.cat_feature_indices = cfg["cat_feature_indices"]
    ensemble.is_fitted = True

    return ensemble, ckpt


def predict_ensemble(df, model_name, checkpoint_path, config_path="feature_columns.json"):
    """Run ensemble inference on new data. Returns DataFrame with published columns + predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble, ckpt = load_ensemble(checkpoint_path, device)

    # Reconstruct scaler from saved params
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = ckpt["scaler_scale"] ** 2
    scaler.n_features_in_ = len(ckpt["scaler_mean"])
    encoders = ckpt["encoders"]

    # Prepare dual-format features using saved encoders/scaler
    X_cat, X_mlp, _, _, _, _, _, _ = prepare_ensemble_data(
        df, model_name, config_path, target_column="__none__",
        fit=False, encoders=encoders, scaler=scaler,
    )

    # Ensemble prediction (weighted probability average)
    proba = ensemble.predict_proba(X_cat, X_mlp)
    preds = np.argmax(proba, axis=1)
    confidence = proba.max(axis=1)

    # Decode labels back to strings if label encoder was used
    if "__label_encoder__" in encoders:
        pred_labels = encoders["__label_encoder__"].inverse_transform(preds)
    else:
        pred_labels = preds

    # Build output DataFrame
    result_df = df.copy()
    result_df["PREDICTION"] = pred_labels
    result_df["PREDICTION_CONFIDENCE"] = np.round(confidence, 4)
    result_df["PredictionDate"] = datetime.now().strftime("%Y-%m-%d")

    # Filter to published columns from feature_columns.json
    feature_config = load_feature_config(config_path)
    pub_cols = feature_config[model_name]["published_columns"]
    out_cols = [c for c in pub_cols if c in result_df.columns]
    if "PREDICTION_CONFIDENCE" not in out_cols:
        out_cols.append("PREDICTION_CONFIDENCE")

    return result_df[out_cols]


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def _generate_synthetic(model_name, config_path, target_column, n=500):
    """Generate synthetic demo data matching feature_columns.json schema."""
    config = load_feature_config(config_path)
    cfg = config[model_name]
    np.random.seed(42)
    data = {}
    for col in cfg["dummy_feature_columns"]:
        cats = [f"{col}_c{i}" for i in range(np.random.randint(3, 8))]
        data[col] = np.random.choice(cats, size=n)
    for col in cfg["additional_feature_columns"]:
        data[col] = np.random.randint(0, 2, size=n)
    data[target_column] = np.random.choice(
        ["OnTime", "Late"], size=n, p=[0.7, 0.3]
    )
    return pd.DataFrame(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train CatBoost (primary) + MLP (secondary) Ensemble"
    )
    parser.add_argument(
        "--model", type=str, default="dccs",
        choices=["etac", "dccs", "csdt"],
        help="Model target (default: dccs)",
    )
    parser.add_argument("--data", type=str, default=None, help="CSV file path")
    parser.add_argument("--sql", type=str, default=None, help="SQL query")
    parser.add_argument(
        "--target", type=str, default="MM_Determination_Completion_Status",
        help="Target column name",
    )
    parser.add_argument("--config", type=str, default="feature_columns.json")
    parser.add_argument("--catboost-weight", type=float, default=0.3)
    parser.add_argument("--mlp-weight", type=float, default=0.7)
    parser.add_argument(
        "--cv-only", action="store_true",
        help="Only run cross-validation, skip final model training",
    )
    args = parser.parse_args()

    # --- Load data ---
    if args.sql:
        df = load_from_sql(args.sql, args.model)
    elif args.data:
        df = load_from_csv(args.data)
    else:
        print("No data source provided — using synthetic demo data...")
        df = _generate_synthetic(args.model, args.config, args.target)

    if len(df) > MAX_SAMPLE_ROWS:
        df = df.sample(n=MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)

    print(f"\nDataset: {len(df)} rows")
    print_feature_summary(args.model, args.config)

    # --- Cross-validation ---
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)
    cv = cross_validate_ensemble(
        df, args.model, args.config, args.target,
        catboost_weight=args.catboost_weight,
        mlp_weight=args.mlp_weight,
    )

    # --- Train final ensemble on full data ---
    if not args.cv_only:
        print("\n" + "=" * 60)
        print("TRAINING FINAL ENSEMBLE ON FULL DATASET")
        print("=" * 60)
        ensemble, enc = train_and_save_ensemble(
            df, args.model, args.config, args.target,
            catboost_weight=args.catboost_weight,
            mlp_weight=args.mlp_weight,
        )
        print("\nDone! To predict on new data:")
        print(f"  from ensemble_train import predict_ensemble, load_from_csv")
        print(f"  df = load_from_csv('new_data.csv')")
        print(f"  results = predict_ensemble(df, '{args.model}', 'ensemble_model_{args.model}.pt')")
        print(f"  results.to_csv('predictions.csv', index=False)")
