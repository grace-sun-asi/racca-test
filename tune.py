"""
Hyperparameter Tuning — CatBoost, MLP, and Ensemble weights via Optuna with tqdm.

Usage:
    python tune.py --model dccs --data DCCS_Train_Data.csv --target MM_Determination_Completion_Status
    python tune.py --model dccs --data DCCS_Train_Data.csv --target MM_Determination_Completion_Status --component catboost
    python tune.py --model dccs --data DCCS_Train_Data.csv --target MM_Determination_Completion_Status --component mlp
    python tune.py --model dccs --data DCCS_Train_Data.csv --target MM_Determination_Completion_Status --component ensemble
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier, Pool
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, Any

from model import FlexibleMLP
from train import train_mlp


# =============================================================================
# CATBOOST TUNING
# =============================================================================

def tune_catboost(
    data: Dict[str, Any],
    n_trials: int = 30,
    n_folds: int = 3,
    metric: str = "accuracy",
) -> Dict[str, Any]:
    """Tune CatBoost hyperparameters with Optuna + tqdm progress."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_cat, y = data["X_cat"], data["y"]
    cat_indices = data["cat_indices"]

    print(f"\nTuning CatBoost | {len(y)} samples | {n_trials} trials | {n_folds} folds")
    trial_times = []
    start = time.time()
    pbar = tqdm(total=n_trials, desc="CatBoost Tuning", unit="trial")

    def objective(trial):
        t0 = time.time()
        params = {
            "iterations": trial.suggest_int("iterations", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", ["Balanced", "SqrtBalanced", "None"]
            ),
            "random_seed": 42, "verbose": 0,
            "eval_metric": "Accuracy", "early_stopping_rounds": 50,
        }
        if params["auto_class_weights"] == "None":
            params["auto_class_weights"] = None

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in skf.split(X_cat, y):
            cb = CatBoostClassifier(**params)
            cb.fit(
                Pool(X_cat[tr_idx], y[tr_idx], cat_features=cat_indices),
                eval_set=Pool(X_cat[va_idx], y[va_idx], cat_features=cat_indices),
                verbose=0,
            )
            preds = cb.predict(X_cat[va_idx]).flatten()
            if metric == "accuracy":
                scores.append(accuracy_score(y[va_idx], preds))
            else:
                scores.append(f1_score(y[va_idx], preds, average="weighted"))

        elapsed = time.time() - t0
        trial_times.append(elapsed)
        avg = np.mean(trial_times)
        remaining = avg * (n_trials - trial.number - 1)

        score = np.mean(scores)
        pbar.update(1)
        pbar.set_postfix({
            "score": f"{score:.4f}",
            "best": f"{study.best_value:.4f}" if study.trials else "N/A",
            "ETA": f"{remaining/60:.1f}m",
        })
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    pbar.close()

    best = study.best_trial.params.copy()
    if best.get("auto_class_weights") == "None":
        best["auto_class_weights"] = None

    total = time.time() - start
    print(f"\nDone in {total/60:.1f}min | Best {metric}: {study.best_value:.4f}")
    print(f"Best params: {best}")

    return {"best_params": best, "best_score": study.best_value, "study": study}


# =============================================================================
# MLP TUNING
# =============================================================================

def tune_mlp(
    data: Dict[str, Any],
    n_trials: int = 30,
    n_folds: int = 3,
    metric: str = "accuracy",
) -> Dict[str, Any]:
    """Tune MLP architecture and training hyperparameters with Optuna + tqdm."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_mlp, y = data["X_mlp"], data["y"]
    n_classes = len(np.unique(y))
    input_dim = X_mlp.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nTuning MLP | {len(y)} samples | {input_dim} features | {n_trials} trials | {n_folds} folds")
    trial_times = []
    start = time.time()
    pbar = tqdm(total=n_trials, desc="MLP Tuning", unit="trial")

    def objective(trial):
        t0 = time.time()

        # Architecture
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_dims = [trial.suggest_int(f"dim_{i}", 32, 512, log=True) for i in range(n_layers)]
        activation = trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu", "selu"])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])

        # Training
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs = trial.suggest_int("epochs", 50, 200)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        for tr_idx, va_idx in skf.split(X_mlp, y):
            model = FlexibleMLP(
                input_dim=input_dim, hidden_dims=hidden_dims,
                output_dim=n_classes, activation=activation,
                dropout=dropout, batch_norm=batch_norm,
            )
            model = train_mlp(
                model, X_mlp[tr_idx], y[tr_idx],
                X_mlp[va_idx], y[va_idx],
                epochs=epochs, lr=lr, batch_size=batch_size, device=device,
            )
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X_mlp[va_idx]).to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            if metric == "accuracy":
                scores.append(accuracy_score(y[va_idx], preds))
            else:
                scores.append(f1_score(y[va_idx], preds, average="weighted"))

        elapsed = time.time() - t0
        trial_times.append(elapsed)
        avg = np.mean(trial_times)
        remaining = avg * (n_trials - trial.number - 1)

        score = np.mean(scores)
        pbar.update(1)
        pbar.set_postfix({
            "score": f"{score:.4f}",
            "best": f"{study.best_value:.4f}" if study.trials else "N/A",
            "ETA": f"{remaining/60:.1f}m",
        })
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    pbar.close()

    # Reconstruct best params
    bp = study.best_trial.params
    n_layers = bp["n_layers"]
    best_params = {
        "hidden_dims": [bp[f"dim_{i}"] for i in range(n_layers)],
        "activation": bp["activation"],
        "dropout": bp["dropout"],
        "batch_norm": bp["batch_norm"],
        "lr": bp["lr"],
        "batch_size": bp["batch_size"],
        "epochs": bp["epochs"],
    }

    total = time.time() - start
    print(f"\nDone in {total/60:.1f}min | Best {metric}: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    return {"best_params": best_params, "best_score": study.best_value, "study": study}


# =============================================================================
# ENSEMBLE WEIGHT TUNING
# =============================================================================

def tune_ensemble_weights(
    data: Dict[str, Any],
    catboost_params: Dict = None,
    mlp_params: Dict = None,
    n_trials: int = 20,
    n_folds: int = 3,
    metric: str = "accuracy",
) -> Dict[str, Any]:
    """Tune the CatBoost/MLP ensemble weight with Optuna + tqdm."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_cat, X_mlp, y = data["X_cat"], data["X_mlp"], data["y"]
    cat_indices = data["cat_indices"]
    n_classes = len(np.unique(y))
    input_dim = X_mlp.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cb_params = catboost_params or {
        "iterations": 1500, "learning_rate": 0.03, "depth": 7,
        "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": 0,
        "eval_metric": "Accuracy", "auto_class_weights": "Balanced",
        "early_stopping_rounds": 50,
    }
    mlp_cfg = mlp_params or {
        "hidden_dims": [min(256, input_dim * 2), min(128, input_dim), 64],
        "activation": "relu", "dropout": 0.2, "batch_norm": True,
        "lr": 1e-3, "batch_size": 32, "epochs": 100,
    }

    print(f"\nTuning ensemble weights | {n_trials} trials | {n_folds} folds")
    pbar = tqdm(total=n_trials, desc="Ensemble Weight Tuning", unit="trial")

    def objective(trial):
        w = trial.suggest_float("catboost_weight", 0.1, 0.9)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        for tr_idx, va_idx in skf.split(X_mlp, y):
            # CatBoost
            cb = CatBoostClassifier(**cb_params)
            cb.fit(
                Pool(X_cat[tr_idx], y[tr_idx], cat_features=cat_indices),
                eval_set=Pool(X_cat[va_idx], y[va_idx], cat_features=cat_indices),
                verbose=0,
            )
            proba_cat = cb.predict_proba(X_cat[va_idx])

            # MLP
            mlp = FlexibleMLP(
                input_dim=input_dim, hidden_dims=mlp_cfg["hidden_dims"],
                output_dim=n_classes, activation=mlp_cfg["activation"],
                dropout=mlp_cfg["dropout"], batch_norm=mlp_cfg["batch_norm"],
            )
            mlp = train_mlp(
                mlp, X_mlp[tr_idx], y[tr_idx], X_mlp[va_idx], y[va_idx],
                epochs=mlp_cfg["epochs"], lr=mlp_cfg["lr"],
                batch_size=mlp_cfg["batch_size"], device=device,
            )
            mlp.eval()
            with torch.no_grad():
                proba_mlp = torch.softmax(
                    mlp(torch.FloatTensor(X_mlp[va_idx]).to(device)), dim=1
                ).cpu().numpy()

            # Ensemble
            proba = w * proba_cat + (1 - w) * proba_mlp
            preds = np.argmax(proba, axis=1)
            if metric == "accuracy":
                scores.append(accuracy_score(y[va_idx], preds))
            else:
                scores.append(f1_score(y[va_idx], preds, average="weighted"))

        score = np.mean(scores)
        pbar.update(1)
        pbar.set_postfix({"weight": f"{w:.2f}", "score": f"{score:.4f}"})
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    pbar.close()

    best_weight = study.best_trial.params["catboost_weight"]
    print(f"\nBest catboost_weight: {best_weight:.3f} | {metric}: {study.best_value:.4f}")

    return {"catboost_weight": best_weight, "best_score": study.best_value, "study": study}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    from data_loader import load_from_csv, prepare_data, MAX_SAMPLE_ROWS

    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument("--model", type=str, default="dccs", choices=["etac", "dccs", "csdt"])
    parser.add_argument("--data", type=str, required=True, help="CSV file path")
    parser.add_argument("--target", type=str, default="MM_Determination_Completion_Status")
    parser.add_argument("--config", type=str, default="feature_columns.json")
    parser.add_argument("--component", type=str, default="all",
                        choices=["catboost", "mlp", "ensemble", "all"])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "f1"])
    args = parser.parse_args()

    df = load_from_csv(args.data)
    if len(df) > MAX_SAMPLE_ROWS:
        df = df.sample(n=MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)

    data = prepare_data(df, args.model, args.target, args.config)
    print(f"Dataset: {len(data['y'])} samples | {len(np.unique(data['y']))} classes")

    results = {}

    if args.component in ("catboost", "all"):
        results["catboost"] = tune_catboost(data, args.trials, args.folds, args.metric)

    if args.component in ("mlp", "all"):
        results["mlp"] = tune_mlp(data, args.trials, args.folds, args.metric)

    if args.component in ("ensemble", "all"):
        cb_params = results.get("catboost", {}).get("best_params")
        mlp_params = results.get("mlp", {}).get("best_params")
        results["ensemble"] = tune_ensemble_weights(
            data, catboost_params=cb_params, mlp_params=mlp_params,
            n_trials=args.trials, n_folds=args.folds, metric=args.metric,
        )

    # Summary
    print(f"\n{'='*50}")
    print("TUNING SUMMARY")
    print(f"{'='*50}")
    for component, res in results.items():
        score = res.get("best_score", "N/A")
        print(f"  {component}: {score:.4f}" if isinstance(score, float) else f"  {component}: {score}")
    print(f"{'='*50}")
