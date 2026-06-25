"""
Training and Evaluation with K-Fold Cross-Validation
-----------------------------------------------------
Provides training loop, evaluation metrics, and stratified k-fold CV
for the FlexibleMLP classifier.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy

from model import FlexibleMLP


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (loss, y_true, y_pred)."""
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

            running_loss += loss.item()
            n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    return avg_loss, y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Full training loop with optional validation monitoring.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader or None
        Validation data loader. If None, no validation metrics are tracked.
    config : dict
        Training configuration with keys:
        - epochs (int): Number of training epochs.
        - lr (float): Learning rate.
        - weight_decay (float): L2 regularization.
        - optimizer (str): 'adam', 'sgd', or 'adamw'.
        - scheduler (str or None): 'step', 'cosine', 'plateau', or None.
        - patience (int): Early stopping patience (0 = disabled).
    device : torch.device
        Device to train on.

    Returns
    -------
    dict with keys: 'model_state', 'train_losses', 'val_losses', 'val_accuracy', 'best_epoch'
    """
    epochs = config.get("epochs", 100)
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 0.0)
    opt_name = config.get("optimizer", "adam")
    sched_name = config.get("scheduler", None)
    patience = config.get("patience", 0)

    # --- Optimizer ---
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'")

    # --- Scheduler ---
    scheduler = None
    if sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

    # --- Loss ---
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    model.to(device)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validation
        if val_loader is not None:
            val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_acc = accuracy_score(y_true, y_pred)
            val_accuracies.append(val_acc)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience > 0 and patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (best: {best_epoch + 1})")
                break

            # Scheduler step
            if scheduler is not None:
                if sched_name == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        else:
            if scheduler is not None and sched_name != "plateau":
                scheduler.step()

    # Restore best model if early stopping was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "model_state": model.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracy": val_accuracies,
        "best_epoch": best_epoch,
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    n_folds: int = 5,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Stratified K-Fold Cross-Validation for the FlexibleMLP.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Integer class labels.
    model_config : dict
        Passed to FlexibleMLP constructor. Keys:
        - hidden_dims, activation, dropout, batch_norm
        (input_dim and output_dim are inferred from data)
    train_config : dict
        Passed to train_model. Keys:
        - epochs, lr, weight_decay, optimizer, scheduler, patience
    n_folds : int
        Number of cross-validation folds.
    batch_size : int
        Batch size for data loaders.
    device : torch.device or None
        Device (auto-detected if None).
    verbose : bool
        Print per-fold results.

    Returns
    -------
    dict with keys:
        'fold_accuracies', 'fold_f1_scores', 'mean_accuracy', 'std_accuracy',
        'mean_f1', 'std_f1', 'fold_reports'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X.shape[1]
    output_dim = len(np.unique(y))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_f1_scores = []
    fold_reports = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        # Create data loaders
        X_train_fold = torch.FloatTensor(X[train_idx])
        y_train_fold = torch.LongTensor(y[train_idx])
        X_val_fold = torch.FloatTensor(X[val_idx])
        y_val_fold = torch.LongTensor(y[val_idx])

        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Instantiate fresh model for each fold
        model = FlexibleMLP(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [128, 64, 32]),
            output_dim=output_dim,
            activation=model_config.get("activation", "relu"),
            dropout=model_config.get("dropout", 0.0),
            batch_norm=model_config.get("batch_norm", False),
        )

        # Train
        results = train_model(model, train_loader, val_loader, train_config, device)

        # Final evaluation on validation fold
        criterion = nn.CrossEntropyLoss()
        model.to(device)
        _, y_true, y_pred = evaluate(model, val_loader, criterion, device)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(y_true, y_pred, output_dict=True)

        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        fold_reports.append(report)

        if verbose:
            print(f"  Accuracy: {acc:.4f} | F1 (weighted): {f1:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Cross-Validation Results ({n_folds} folds):")
        print(f"  Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"  F1 Score: {mean_f1:.4f} +/- {std_f1:.4f}")
        print(f"{'='*50}")

    return {
        "fold_accuracies": fold_accuracies,
        "fold_f1_scores": fold_f1_scores,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "fold_reports": fold_reports,
    }
