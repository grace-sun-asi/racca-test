"""
CatBoost Hyperparameter Tuning with Optuna
--------------------------------------------
Tunes CatBoost classifier using Bayesian optimization (TPE sampler).
Uses stratified K-fold CV as the objective metric.

Usage:
    python catboost_tuning.py --model dccs --data DCCS_Train_Data.csv --target MM_Determination_Completion_Status
    python catboost_tuning.py --model dccs --data DCCS_Train_Data.csv --target MM_Determination_Completion_Status --trials 50
    python catboost_tuning.py --model dccs --data DCCS_Train_Data.csv --target MM_Determination_Completion_Status --train-best
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier, Pool
from typing import Dict, Any

from data_loader import (
    load_feature_config, load_from_sql, load_from_csv,
    build_derived_columns, print_feature_summary,
    MAX_SAMPLE_ROWS,
)
from ensemble_train import prepare_ensemble_data

OUTPUT_DIR = Path("output/ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def tune_catboost(
    df: pd.DataFrame,
    model_name: str,
    target_column: str = "MM_Determination_Completion_Status",
    config_path: str = "feature_columns.json",
    n_trials: int = 30,
    n_folds: int = 3,
    metric: str = "accuracy",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Tune CatBoost hyperparameters using Optuna Bayesian optimization.

    Search space:
    - iterations: 200-2000
    - learning_rate: 0.01-0.3
    - depth: 3-10
    - l2_leaf_reg: 1-10
    - bagging_temperature: 0-5
    - random_strength: 0-5
    - border_count: 32-255
    - min_data_in_leaf: 1-50

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    model_name : str
        Model target (dccs, etac, csdt).
    target_column : str
        Target column name.
    config_path : str
        Path to feature_columns.json.
    n_trials : int
        Number of Optuna trials to run.
    n_folds : int
        Number of CV folds per trial.
    metric : str
        'accuracy' or 'f1'.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with 'best_params', 'best_score', 'study'
    """
    try:
        import optuna
        optuna.logging.set_verbosity(
            optuna.logging.WARNING if not verbose else optuna.logging.INFO
        )
    except ImportError:
        raise ImportError("Optuna required. Install with: pip install optuna")

    # Prepare data (CatBoost format)
    X_cat, _, y, cat_indices, _, _, _, cat_names = prepare_ensemble_data(
        df, model_name, config_path, target_column, fit=True
    )

    n_classes = len(np.unique(y))
    if verbose:
        print(f"\nTuning CatBoost for '{model_name}'")
        print(f"  Samples: {len(y)} | Features: {X_cat.shape[1]} | Classes: {n_classes}")
        print(f"  Trials: {n_trials} | Folds: {n_folds} | Metric: {metric}")
        print()

    def objective(trial):
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
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "Accuracy",
            "early_stopping_rounds": 50,
        }

        # Fix "None" string to actual None
        if params["auto_class_weights"] == "None":
            params["auto_class_weights"] = None

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(X_cat, y):
            X_tr, X_va = X_cat[train_idx], X_cat[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(
                Pool(X_tr, y_tr, cat_features=cat_indices),
                eval_set=Pool(X_va, y_va, cat_features=cat_indices),
                verbose=0,
            )

            preds = model.predict(X_va).flatten()
            if metric == "accuracy":
                scores.append(accuracy_score(y_va, preds))
            else:
                scores.append(f1_score(y_va, preds, average="weighted"))

        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_params = best.params.copy()
    if best_params.get("auto_class_weights") == "None":
        best_params["auto_class_weights"] = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"BEST CATBOOST PARAMS (trial {best.number})")
        print(f"  {metric}: {best.value:.4f}")
        print(f"{'='*60}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")

    return {"best_params": best_params, "best_score": best.value, "study": study}


def train_best_catboost(
    df: pd.DataFrame,
    model_name: str,
    best_params: Dict[str, Any],
    target_column: str = "MM_Determination_Completion_Status",
    config_path: str = "feature_columns.json",
    save_path: str = None,
    verbose: bool = True,
) -> CatBoostClassifier:
    """
    Train CatBoost with the best params from tuning on full dataset and save.
    """
    X_cat, _, y, cat_indices, _, _, _, cat_names = prepare_ensemble_data(
        df, model_name, config_path, target_column, fit=True
    )

    # Clean up params for training
    train_params = best_params.copy()
    train_params["verbose"] = 50 if verbose else 0
    train_params["random_seed"] = 42
    train_params["eval_metric"] = "Accuracy"
    # Remove early_stopping for full training (no eval set)
    train_params.pop("early_stopping_rounds", None)

    model = CatBoostClassifier(**train_params)
    model.fit(Pool(X_cat, y, cat_features=cat_indices))

    if save_path is None:
        save_path = str(OUTPUT_DIR / f"catboost_{model_name}_tuned.cbm")

    model.save_model(save_path)
    if verbose:
        print(f"\nSaved tuned CatBoost to: {save_path}")
        print(f"Feature importance (top 10):")
        importance = model.get_feature_importance()
        fi = sorted(zip(cat_names, importance), key=lambda x: x[1], reverse=True)
        for name, imp in fi[:10]:
            print(f"  {name}: {imp:.2f}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CatBoost Hyperparameter Tuning")
    parser.add_argument("--model", type=str, default="dccs", choices=["etac", "dccs", "csdt"])
    parser.add_argument("--data", type=str, required=True, help="CSV file path")
    parser.add_argument("--target", type=str, default="MM_Determination_Completion_Status")
    parser.add_argument("--config", type=str, default="feature_columns.json")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--folds", type=int, default=3, help="CV folds per trial")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "f1"])
    parser.add_argument("--train-best", action="store_true", help="Train final model with best params")
    args = parser.parse_args()

    # Load data
    df = load_from_csv(args.data)
    if len(df) > MAX_SAMPLE_ROWS:
        df = df.sample(n=MAX_SAMPLE_ROWS, random_state=42).reset_index(drop=True)

    print(f"Dataset: {len(df)} rows")
    print_feature_summary(args.model, args.config)

    # Run tuning
    results = tune_catboost(
        df, args.model, args.target, args.config,
        n_trials=args.trials, n_folds=args.folds, metric=args.metric,
    )

    # Optionally train final model with best params
    if args.train_best:
        print("\n" + "=" * 60)
        print("TRAINING FINAL MODEL WITH BEST PARAMS")
        print("=" * 60)
        model = train_best_catboost(
            df, args.model, results["best_params"], args.target, args.config,
        )

    # Print command to use these params in ensemble_train.py
    print("\n\nTo use these params in ensemble_train.py, pass them via code:")
    print("  catboost_params = {")
    for k, v in results["best_params"].items():
        print(f'      "{k}": {repr(v)},')
    print("  }")
