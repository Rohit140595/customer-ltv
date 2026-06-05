"""
Model training, evaluation, and SHAP feature selection.

Responsibilities:
  - Prepare feature matrix and target for training (drop high-missing, encode cats)
  - SHAP-based feature selection: top N features by mean |SHAP| value
  - Optuna hyperparameter tuning with TimeSeriesSplit cross-validation
  - Train final XGBoost regression model
  - Evaluate: RMSE, MAE, R², and top-decile lift (how well we rank high-LTV customers)
  - Persist model artifacts to models/

Top-decile lift is the primary business metric: if we score the top 10% of customers
by predicted LTV, how much of the actual total revenue do they account for?
A lift of 3x means the top decile generates 3x their proportional share of revenue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    drop_missing_threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare feature matrix for training.

    Steps:
      1. Drop columns with > drop_missing_threshold missing values.
      2. Drop constant columns (zero variance).
      3. Return cleaned X, y, and list of dropped column names.

    Args:
        X:                      Feature matrix.
        y:                      Target series (ltv_target).
        drop_missing_threshold: Columns with more missing than this are dropped.

    Returns:
        (X_clean, y_clean, dropped_cols)
    """
    raise NotImplementedError


def select_features_shap(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 80,
    cache_path: str | None = None,
) -> list[str]:
    """
    Select the top N features by mean absolute SHAP value.

    Fits a lightweight XGBoost model, computes SHAP values, ranks features
    by mean |SHAP|, and returns the top N. Result is cached to disk so
    subsequent runs load instantly.

    Args:
        X_train:    Training feature matrix.
        y_train:    Training target.
        top_n:      Number of features to keep.
        cache_path: Path to JSON cache file. If exists, loads from cache.

    Returns:
        List of selected feature names.
    """
    raise NotImplementedError


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Tune XGBoost hyperparameters with Optuna and TimeSeriesSplit CV.

    Each CV fold's validation data is strictly later than its training data,
    consistent with the temporal ordering of LTV predictions in production.

    Optimises for RMSE on the validation fold.

    Args:
        X_train:      Training feature matrix.
        y_train:      Training target.
        n_trials:     Number of Optuna trials.
        cv_folds:     Number of TimeSeriesSplit folds.
        random_state: Random seed for reproducibility.

    Returns:
        Dict of best hyperparameters.
    """
    raise NotImplementedError


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> object:
    """
    Train the final XGBoost regression model with tuned hyperparameters.

    Args:
        X_train:     Training feature matrix.
        y_train:     Training target.
        best_params: Hyperparameters from tune_hyperparameters().

    Returns:
        Fitted XGBoost model.
    """
    raise NotImplementedError


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate model on the held-out test set.

    Metrics:
        rmse          : Root mean squared error.
        mae           : Mean absolute error.
        r2            : R² (coefficient of determination).
        top_decile_lift: Revenue captured by top 10% of predicted LTV customers,
                         divided by their proportional share (i.e. lift over random).

    Args:
        model:  Fitted model with a predict() method.
        X_test: Test feature matrix.
        y_test: True LTV targets.

    Returns:
        Dict of metric name → value.
    """
    raise NotImplementedError


def save_model(model, feature_names: list[str], path: str) -> None:
    """Persist model and feature names to disk as a pickle."""
    raise NotImplementedError


def load_model(path: str) -> tuple:
    """Load model and feature names from disk."""
    raise NotImplementedError
