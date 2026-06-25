"""
Hyperparameter Tuning Module
-----------------------------
Provides grid search, random search, and Optuna-based Bayesian optimization
over the full model + training configuration space.

The search space is completely flexible: architecture depth, width, activation,
regularization, optimizer, scheduler, and learning rate are all tunable.
"""

import numpy as np
import itertools
import random
from typing import Dict, Any, List, Optional
from copy import deepcopy

from train import cross_validate


# =============================================================================
# SEARCH SPACE DEFINITION
# =============================================================================

# Default search space — modify or extend as needed
DEFAULT_SEARCH_SPACE = {
    # --- Architecture ---
    "hidden_dims": [
        [64, 32],              # 2 layers (shallow)
        [128, 64, 32],         # 3 layers (baseline)
        [256, 128, 64],        # 3 layers (wider)
        [256, 128, 64, 32],    # 4 layers (deeper)
        [512, 256, 128, 64],   # 4 layers (wide + deep)
        [128, 128, 128],       # 3 layers (uniform)
    ],
    "activation": ["relu", "leaky_relu", "gelu", "selu"],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
    "batch_norm": [True, False],

    # --- Training ---
    "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "optimizer": ["adam", "adamw", "sgd"],
    "scheduler": [None, "cosine", "step", "plateau"],
    "batch_size": [16, 32, 64, 128],
    "epochs": [50, 100, 150, 200],
    "patience": [10, 15, 20, 0],  # 0 = no early stopping
}


def _split_config(params: Dict[str, Any]) -> tuple:
    """Split a flat param dict into (model_config, train_config, batch_size)."""
    model_keys = {"hidden_dims", "activation", "dropout", "batch_norm"}
    train_keys = {"lr", "weight_decay", "optimizer", "scheduler", "epochs", "patience"}

    model_config = {k: v for k, v in params.items() if k in model_keys}
    train_config = {k: v for k, v in params.items() if k in train_keys}
    batch_size = params.get("batch_size", 32)

    return model_config, train_config, batch_size


# =============================================================================
# GRID SEARCH
# =============================================================================

def grid_search(
    X: np.ndarray,
    y: np.ndarray,
    search_space: Optional[Dict[str, List]] = None,
    n_folds: int = 3,
    metric: str = "accuracy",
    device=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Exhaustive grid search over all combinations in the search space.

    Warning: Can be very slow for large spaces. Use a reduced space or
    random_search for larger explorations.

    Returns
    -------
    dict with 'best_params', 'best_score', 'all_results'
    """
    if search_space is None:
        # Use a smaller default for grid search to be practical
        search_space = {
            "hidden_dims": [[128, 64, 32], [256, 128, 64]],
            "activation": ["relu", "gelu"],
            "dropout": [0.0, 0.2],
            "batch_norm": [True, False],
            "lr": [1e-3, 5e-4],
            "weight_decay": [0.0, 1e-4],
            "optimizer": ["adam"],
            "scheduler": ["cosine"],
            "batch_size": [32],
            "epochs": [100],
            "patience": [15],
        }

    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))

    print(f"Grid Search: {len(combinations)} total combinations")

    best_score = -1.0
    best_params = None
    all_results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        model_config, train_config, batch_size = _split_config(params)

        if verbose:
            print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")

        cv_results = cross_validate(
            X=X, y=y,
            model_config=model_config,
            train_config=train_config,
            n_folds=n_folds,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )

        score = cv_results["mean_accuracy"] if metric == "accuracy" else cv_results["mean_f1"]

        all_results.append({"params": params, "score": score, "std": cv_results["std_accuracy"]})

        if score > best_score:
            best_score = score
            best_params = params
            if verbose:
                print(f"  -> New best! {metric}={score:.4f}")

    print(f"\nBest {metric}: {best_score:.4f}")
    print(f"Best params: {best_params}")

    return {"best_params": best_params, "best_score": best_score, "all_results": all_results}


# =============================================================================
# RANDOM SEARCH
# =============================================================================

def random_search(
    X: np.ndarray,
    y: np.ndarray,
    search_space: Optional[Dict[str, List]] = None,
    n_trials: int = 20,
    n_folds: int = 3,
    metric: str = "accuracy",
    device=None,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Random search — samples n_trials random configurations from the space.
    More efficient than grid search for high-dimensional spaces.

    Returns
    -------
    dict with 'best_params', 'best_score', 'all_results'
    """
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    random.seed(seed)
    np.random.seed(seed)

    print(f"Random Search: {n_trials} trials from search space")

    best_score = -1.0
    best_params = None
    all_results = []

    for trial in range(n_trials):
        # Sample one value from each dimension
        params = {k: random.choice(v) for k, v in search_space.items()}
        model_config, train_config, batch_size = _split_config(params)

        if verbose:
            print(f"\n[Trial {trial+1}/{n_trials}] {params}")

        cv_results = cross_validate(
            X=X, y=y,
            model_config=model_config,
            train_config=train_config,
            n_folds=n_folds,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )

        score = cv_results["mean_accuracy"] if metric == "accuracy" else cv_results["mean_f1"]
        all_results.append({"params": params, "score": score, "std": cv_results["std_accuracy"]})

        if score > best_score:
            best_score = score
            best_params = params
            if verbose:
                print(f"  -> New best! {metric}={score:.4f}")

    print(f"\nBest {metric}: {best_score:.4f}")
    print(f"Best params: {best_params}")

    return {"best_params": best_params, "best_score": best_score, "all_results": all_results}


# =============================================================================
# OPTUNA-BASED BAYESIAN OPTIMIZATION (optional dependency)
# =============================================================================

def optuna_search(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_folds: int = 3,
    metric: str = "accuracy",
    device=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Bayesian hyperparameter optimization using Optuna.
    Requires: pip install optuna

    This is the most sample-efficient search method — uses Tree-structured
    Parzen Estimator (TPE) to focus on promising regions of the search space.

    The search space here is CONTINUOUS and CONDITIONAL, making it much more
    flexible than grid/random search.

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
        raise ImportError(
            "Optuna is required for Bayesian optimization. Install with: pip install optuna"
        )

    def objective(trial):
        # --- Architecture search (flexible depth + width) ---
        n_layers = trial.suggest_int("n_layers", 1, 5)
        hidden_dims = []
        for i in range(n_layers):
            # Each layer can have a different width, sampled log-uniformly
            dim = trial.suggest_int(f"layer_{i}_dim", 16, 512, log=True)
            hidden_dims.append(dim)

        activation = trial.suggest_categorical(
            "activation", ["relu", "leaky_relu", "gelu", "selu", "elu"]
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])

        # --- Training search ---
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        scheduler_name = trial.suggest_categorical(
            "scheduler", ["none", "cosine", "step", "plateau"]
        )
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        epochs = trial.suggest_int("epochs", 30, 200)
        patience = trial.suggest_int("patience", 5, 30)

        model_config = {
            "hidden_dims": hidden_dims,
            "activation": activation,
            "dropout": dropout,
            "batch_norm": batch_norm,
        }

        train_config = {
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "optimizer": optimizer_name,
            "scheduler": None if scheduler_name == "none" else scheduler_name,
            "patience": patience,
        }

        cv_results = cross_validate(
            X=X, y=y,
            model_config=model_config,
            train_config=train_config,
            n_folds=n_folds,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )

        score = (
            cv_results["mean_accuracy"] if metric == "accuracy" else cv_results["mean_f1"]
        )
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial

    # Reconstruct best params in a usable format
    n_layers = best_trial.params["n_layers"]
    hidden_dims = [best_trial.params[f"layer_{i}_dim"] for i in range(n_layers)]

    best_params = {
        "hidden_dims": hidden_dims,
        "activation": best_trial.params["activation"],
        "dropout": best_trial.params["dropout"],
        "batch_norm": best_trial.params["batch_norm"],
        "lr": best_trial.params["lr"],
        "weight_decay": best_trial.params["weight_decay"],
        "optimizer": best_trial.params["optimizer"],
        "scheduler": (
            None
            if best_trial.params["scheduler"] == "none"
            else best_trial.params["scheduler"]
        ),
        "batch_size": best_trial.params["batch_size"],
        "epochs": best_trial.params["epochs"],
        "patience": best_trial.params["patience"],
    }

    print(f"\nOptuna Best {metric}: {best_trial.value:.4f}")
    print(f"Best params: {best_params}")

    return {"best_params": best_params, "best_score": best_trial.value, "study": study}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import torch
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler

    # Generate sample data
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=15,
        n_classes=3, n_clusters_per_class=2, random_state=42,
    )
    X = StandardScaler().fit_transform(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("RANDOM SEARCH (10 trials)")
    print("=" * 60)
    results = random_search(X, y, n_trials=10, n_folds=3, device=device)

    # Uncomment below to try Optuna (requires: pip install optuna)
    # print("\n" + "=" * 60)
    # print("OPTUNA BAYESIAN SEARCH (30 trials)")
    # print("=" * 60)
    # results = optuna_search(X, y, n_trials=30, n_folds=3, device=device)
