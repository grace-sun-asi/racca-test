"""
Main CLI — Train, Tune, or Predict with the CatBoost + MLP Ensemble.

Usage:
    python main.py train --model dccs --data DCCS_Train_Data.csv --target MM_Determination_Completion_Status
    python main.py tune  --model dccs --data DCCS_Train_Data.csv --target MM_Determination_Completion_Status --component catboost
    python main.py predict --model dccs --data new_data.csv
"""
from __future__ import annotations

import argparse
import numpy as np

from data_loader import load_from_csv, load_from_sql, prepare_data


def cmd_train(args):
    from train import cross_validate_catboost, train_catboost, cross_validate_ensemble, train_ensemble

    df = _load_data(args)
    data = prepare_data(df, args.model, args.target, args.config)
    print(f"Prepared: {len(data['y'])} samples | {len(np.unique(data['y']))} classes")

    if args.ensemble:
        # Full ensemble (CatBoost + MLP) — slower
        cross_validate_ensemble(data, n_folds=args.folds, catboost_weight=args.catboost_weight)
        train_ensemble(data, args.model, catboost_weight=args.catboost_weight)
    else:
        # CatBoost only — fast
        cross_validate_catboost(data, n_folds=args.folds)
        train_catboost(data, args.model)


def cmd_tune(args):
    from tune import tune_catboost, tune_mlp, tune_ensemble_weights

    df = _load_data(args)
    data = prepare_data(df, args.model, args.target, args.config)
    print(f"Prepared: {len(data['y'])} samples | {len(np.unique(data['y']))} classes")

    if args.component in ("catboost", "all"):
        tune_catboost(data, n_trials=args.trials, n_folds=args.folds, metric=args.metric)

    if args.component in ("mlp", "all"):
        tune_mlp(data, n_trials=args.trials, n_folds=args.folds, metric=args.metric)

    if args.component in ("ensemble", "all"):
        tune_ensemble_weights(data, n_trials=args.trials, n_folds=args.folds, metric=args.metric)


def cmd_predict(args):
    from inference import predict

    df = _load_data(args)
    checkpoint = args.checkpoint or f"ensemble_model_{args.model}.pt"
    output_path = args.output or f"predictions_{args.model}.csv"

    results = predict(df, args.model, checkpoint, args.config)
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path} ({len(results)} rows)")


def _load_data(args) -> "pd.DataFrame":
    if args.sql:
        df = load_from_sql(args.sql, args.model)
    elif args.data:
        df = load_from_csv(args.data)
    else:
        raise ValueError("Provide --data or --sql")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RACCA ML Pipeline")
    sub = parser.add_subparsers(dest="command")

    # Shared args
    def add_common(p):
        p.add_argument("--model", type=str, default="dccs", choices=["etac", "dccs", "csdt"])
        p.add_argument("--data", type=str, default=None, help="CSV file path")
        p.add_argument("--sql", type=str, default=None, help="SQL query")
        p.add_argument("--config", type=str, default="feature_columns.json")

    # Train
    p_train = sub.add_parser("train", help="Train ensemble model")
    add_common(p_train)
    p_train.add_argument("--target", type=str, default="MM_Determination_Completion_Status")
    p_train.add_argument("--folds", type=int, default=3)
    p_train.add_argument("--catboost-weight", type=float, default=0.6)
    p_train.add_argument("--ensemble", action="store_true", help="Train full ensemble (CatBoost+MLP). Default: CatBoost only.")

    # Tune
    p_tune = sub.add_parser("tune", help="Hyperparameter tuning")
    add_common(p_tune)
    p_tune.add_argument("--target", type=str, default="MM_Determination_Completion_Status")
    p_tune.add_argument("--component", type=str, default="all",
                        choices=["catboost", "mlp", "ensemble", "all"])
    p_tune.add_argument("--trials", type=int, default=30)
    p_tune.add_argument("--folds", type=int, default=3)
    p_tune.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "f1"])

    # Predict
    p_pred = sub.add_parser("predict", help="Run inference")
    add_common(p_pred)
    p_pred.add_argument("--checkpoint", type=str, default=None)
    p_pred.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()
