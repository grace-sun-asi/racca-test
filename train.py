"""
Training — CatBoost, MLP, and Ensemble training + cross-validation with tqdm.
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

from model import FlexibleMLP

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# MLP TRAINING
# =============================================================================

def train_mlp(
    model: FlexibleMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 15,
    device: torch.device = None,
) -> FlexibleMLP:
    """Train MLP with early stopping. Returns trained model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

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

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_t = torch.FloatTensor(X_val).to(device)
                val_loss = criterion(model(val_t), torch.LongTensor(y_val).to(device)).item()
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
# CROSS-VALIDATION (CatBoost only — fast)
# =============================================================================

def cross_validate_catboost(
    data: Dict[str, Any],
    n_folds: int = 3,
    catboost_params: Dict = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Stratified K-Fold CV for CatBoost only. Fast — no MLP overhead."""
    X_cat, y = data["X_cat"], data["y"]
    cat_indices = data["cat_indices"]
    encoders = data["encoders"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cb_params = catboost_params or {
        "iterations": 500, "learning_rate": 0.05, "depth": 6,
        "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": 0,
        "eval_metric": "Accuracy", "auto_class_weights": "Balanced",
        "early_stopping_rounds": 30,
    }

    accuracies, f1_scores = [], []
    all_y_true, all_y_pred = [], []

    fold_iter = tqdm(
        enumerate(skf.split(X_cat, y)),
        total=n_folds, desc="CV Folds", unit="fold",
    )

    for fold, (tr_idx, va_idx) in fold_iter:
        fold_start = time.time()
        print(f"\n  Fold {fold+1}/{n_folds} — Training CatBoost ({cb_params.get('iterations', 500)} iters)...")

        cb = CatBoostClassifier(**cb_params)
        cb.fit(
            Pool(X_cat[tr_idx], y[tr_idx], cat_features=cat_indices),
            eval_set=Pool(X_cat[va_idx], y[va_idx], cat_features=cat_indices),
            verbose=100,
        )

        preds = cb.predict(X_cat[va_idx]).flatten().astype(int)
        acc = accuracy_score(y[va_idx], preds)
        f1 = f1_score(y[va_idx], preds, average="weighted")
        accuracies.append(acc)
        f1_scores.append(f1)

        all_y_true.extend(y[va_idx])
        all_y_pred.extend(preds)

        fold_elapsed = time.time() - fold_start
        fold_iter.set_postfix({"acc": f"{acc:.4f}", "f1": f"{f1:.4f}", "time": f"{fold_elapsed:.0f}s"})

    # Get class names for reporting
    if "__label_encoder__" in encoders:
        class_names = list(encoders["__label_encoder__"].classes_)
    else:
        class_names = [str(i) for i in range(len(np.unique(y)))]

    if verbose:
        print(f"\n{'='*60}")
        print(f"CV RESULTS — CatBoost ({n_folds} folds)")
        print(f"{'='*60}")
        print(f"  Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
        print(f"  F1:       {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
        print(f"\n  Per-Class Report:")
        print(classification_report(all_y_true, all_y_pred, target_names=class_names, digits=4))
        print(f"  Confusion Matrix:")
        cm = confusion_matrix(all_y_true, all_y_pred)
        # Header
        header = "  Predicted →  " + "  ".join(f"{n:>12}" for n in class_names)
        print(header)
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>12}" for v in row)
            print(f"  Actual {class_names[i]:<12} {row_str}")
        print(f"{'='*60}")

    return {"accuracies": accuracies, "f1_scores": f1_scores}


# =============================================================================
# TRAIN FINAL CATBOOST (full dataset) + SAVE
# =============================================================================

def train_catboost(
    data: Dict[str, Any],
    model_name: str,
    catboost_params: Dict = None,
    save_path: str = None,
    verbose: bool = True,
) -> CatBoostClassifier:
    """Train CatBoost on full dataset and save."""
    X_cat, y = data["X_cat"], data["y"]
    cat_indices = data["cat_indices"]

    cb_params = catboost_params or {
        "iterations": 1500, "learning_rate": 0.03, "depth": 7,
        "l2_leaf_reg": 3.0, "random_seed": 42,
        "verbose": 50 if verbose else 0,
        "eval_metric": "Accuracy", "auto_class_weights": "Balanced",
    }

    if verbose:
        print(f"Training CatBoost on {len(y)} samples, {len(np.unique(y))} classes")

    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(Pool(X_cat, y, cat_features=cat_indices))

    if save_path is None:
        save_path = str(OUTPUT_DIR / f"catboost_{model_name}.cbm")
    cb_model.save_model(save_path)

    # Also save metadata for inference
    meta_path = str(OUTPUT_DIR / f"catboost_{model_name}_meta.pt")
    torch.save({
        "encoders": data["encoders"],
        "scaler_mean": data["scaler"].mean_,
        "scaler_scale": data["scaler"].scale_,
        "cat_feature_names": data["cat_feature_names"],
        "model_name": model_name,
        "n_classes": len(np.unique(y)),
    }, meta_path)

    if verbose:
        print(f"\nSaved: {save_path}")
        print(f"Saved metadata: {meta_path}")

    return cb_model


# =============================================================================
# CROSS-VALIDATION (Ensemble: CatBoost + MLP)
# =============================================================================

def cross_validate_ensemble(
    data: Dict[str, Any],
    n_folds: int = 3,
    catboost_params: Dict = None,
    mlp_epochs: int = 100,
    catboost_weight: float = 0.6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Stratified K-Fold CV comparing CatBoost, MLP, and Ensemble.
    Uses tqdm progress bars for folds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_cat, X_mlp, y = data["X_cat"], data["X_mlp"], data["y"]
    cat_indices = data["cat_indices"]
    mlp_weight = 1.0 - catboost_weight

    n_classes = len(np.unique(y))
    mlp_input_dim = X_mlp.shape[1]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cb_params = catboost_params or {
        "iterations": 500, "learning_rate": 0.05, "depth": 6,
        "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": 0,
        "eval_metric": "Accuracy", "auto_class_weights": "Balanced",
        "early_stopping_rounds": 30,
    }

    results = {"catboost": [], "mlp": [], "ensemble": [], "f1": []}

    fold_iter = tqdm(
        enumerate(skf.split(X_mlp, y)),
        total=n_folds, desc="CV Folds", unit="fold",
    )

    for fold, (tr_idx, va_idx) in fold_iter:
        fold_start = time.time()
        X_cat_tr, X_cat_va = X_cat[tr_idx], X_cat[va_idx]
        X_mlp_tr, X_mlp_va = X_mlp[tr_idx], X_mlp[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # CatBoost
        print(f"\n  Fold {fold+1}/{n_folds} — Training CatBoost ({cb_params.get('iterations', 500)} iters)...")
        cb = CatBoostClassifier(**cb_params)
        cb.fit(
            Pool(X_cat_tr, y_tr, cat_features=cat_indices),
            eval_set=Pool(X_cat_va, y_va, cat_features=cat_indices),
            verbose=100,
        )
        proba_cat = cb.predict_proba(X_cat_va)
        acc_cat = accuracy_score(y_va, np.argmax(proba_cat, axis=1))

        # MLP
        print(f"  Fold {fold+1}/{n_folds} — Training MLP ({mlp_epochs} epochs)...")
        mlp = FlexibleMLP(
            input_dim=mlp_input_dim,
            hidden_dims=[min(256, mlp_input_dim * 2), min(128, mlp_input_dim), 64],
            output_dim=n_classes, activation="relu", dropout=0.2, batch_norm=True,
        )
        mlp = train_mlp(mlp, X_mlp_tr, y_tr, X_mlp_va, y_va, epochs=mlp_epochs, device=device)
        mlp.eval()
        with torch.no_grad():
            proba_mlp = torch.softmax(
                mlp(torch.FloatTensor(X_mlp_va).to(device)), dim=1
            ).cpu().numpy()
        acc_mlp = accuracy_score(y_va, np.argmax(proba_mlp, axis=1))

        # Ensemble
        proba_ens = catboost_weight * proba_cat + mlp_weight * proba_mlp
        pred_ens = np.argmax(proba_ens, axis=1)
        acc_ens = accuracy_score(y_va, pred_ens)
        f1_ens = f1_score(y_va, pred_ens, average="weighted")

        results["catboost"].append(acc_cat)
        results["mlp"].append(acc_mlp)
        results["ensemble"].append(acc_ens)
        results["f1"].append(f1_ens)

        fold_elapsed = time.time() - fold_start
        fold_iter.set_postfix({
            "cb": f"{acc_cat:.4f}", "mlp": f"{acc_mlp:.4f}",
            "ens": f"{acc_ens:.4f}", "time": f"{fold_elapsed:.0f}s",
        })

    if verbose:
        print(f"\n{'='*50}")
        print(f"CV RESULTS ({n_folds} folds)")
        print(f"  CatBoost: {np.mean(results['catboost']):.4f} +/- {np.std(results['catboost']):.4f}")
        print(f"  MLP:      {np.mean(results['mlp']):.4f} +/- {np.std(results['mlp']):.4f}")
        print(f"  Ensemble: {np.mean(results['ensemble']):.4f} +/- {np.std(results['ensemble']):.4f}")
        print(f"  F1 (ens): {np.mean(results['f1']):.4f}")
        print(f"{'='*50}")

    return results


# =============================================================================
# TRAIN FINAL ENSEMBLE (full dataset) + SAVE
# =============================================================================

def train_ensemble(
    data: Dict[str, Any],
    model_name: str,
    catboost_params: Dict = None,
    mlp_epochs: int = 100,
    catboost_weight: float = 0.6,
    save_path: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train CatBoost + MLP on full dataset and save.
    Returns dict with models, config, paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_cat, X_mlp, y = data["X_cat"], data["X_mlp"], data["y"]
    cat_indices = data["cat_indices"]

    n_classes = len(np.unique(y))
    mlp_input_dim = X_mlp.shape[1]

    cb_params = catboost_params or {
        "iterations": 1500, "learning_rate": 0.03, "depth": 7,
        "l2_leaf_reg": 3.0, "random_seed": 42,
        "verbose": 50 if verbose else 0,
        "eval_metric": "Accuracy", "auto_class_weights": "Balanced",
    }

    if verbose:
        print(f"Training ensemble on {len(y)} samples, {n_classes} classes")
        print(f"  CatBoost: {X_cat.shape[1]} features | MLP: {X_mlp.shape[1]} features")

    # Train CatBoost
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(Pool(X_cat, y, cat_features=cat_indices))

    # Train MLP
    mlp_hidden = [min(256, mlp_input_dim * 2), min(128, mlp_input_dim), 64]
    mlp_model = FlexibleMLP(
        input_dim=mlp_input_dim, hidden_dims=mlp_hidden,
        output_dim=n_classes, activation="relu", dropout=0.2, batch_norm=True,
    )
    mlp_model = train_mlp(mlp_model, X_mlp, y, epochs=mlp_epochs, device=device)

    # Save
    if save_path is None:
        save_path = f"ensemble_model_{model_name}.pt"
    cb_path = str(OUTPUT_DIR / f"catboost_{model_name}.cbm")
    cb_model.save_model(cb_path)

    torch.save({
        "mlp_state_dict": mlp_model.state_dict(),
        "mlp_hidden_dims": mlp_hidden,
        "mlp_input_dim": mlp_input_dim,
        "n_classes": n_classes,
        "catboost_path": cb_path,
        "catboost_weight": catboost_weight,
        "encoders": data["encoders"],
        "scaler_mean": data["scaler"].mean_,
        "scaler_scale": data["scaler"].scale_,
        "cat_feature_names": data["cat_feature_names"],
        "mlp_feature_names": data["mlp_feature_names"],
        "model_name": model_name,
    }, save_path)

    if verbose:
        print(f"\nSaved CatBoost -> {cb_path}")
        print(f"Saved ensemble -> {save_path}")

    return {"catboost": cb_model, "mlp": mlp_model, "save_path": save_path}
