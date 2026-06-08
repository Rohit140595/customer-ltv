"""
Two-stage model training, evaluation, and persistence.

Stage 1 — Return classifier (binary):
    Trained on all ~81K customers. Predicts will_return (0/1).
    Primary metric: PR-AUC (imbalanced — only 0.3% return).
    CV: TimeSeriesSplit (temporal ordering must be preserved).

Stage 2 — Spend-tier classifier (3-class):
    Trained on ~197 returners only. Predicts spend_tier (0=Low, 1=Mid, 2=High).
    Primary metric: weighted F1.
    CV: StratifiedKFold (small dataset — need balanced classes per fold).

Both stages use XGBoost and Optuna for hyperparameter tuning.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

TIER_LABELS = {0: "Low", 1: "Mid", 2: "High"}


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    drop_missing_threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Clean the feature matrix before training.

    Drops:
      - Columns with > drop_missing_threshold fraction missing.
      - Constant columns (only one unique non-null value).

    Args:
        X:                      Feature matrix (customer_unique_id excluded).
        y:                      Target series.
        drop_missing_threshold: Columns above this missing rate are dropped.

    Returns:
        (X_clean, y_clean, dropped_cols)
    """
    drop_cols = []

    # Drop high-missing columns
    missing_rate = X.isnull().mean()
    high_missing  = missing_rate[missing_rate > drop_missing_threshold].index.tolist()
    drop_cols.extend(high_missing)

    # Drop constant columns
    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    drop_cols.extend([c for c in constant if c not in drop_cols])

    X_clean = X.drop(columns=drop_cols)
    if drop_cols:
        print(f"Dropped {len(drop_cols)} columns: {drop_cols}")

    return X_clean, y, drop_cols


# ── SHAP feature selection ────────────────────────────────────────────────────

def select_features_shap(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 15,
    cache_path: str | None = None,
) -> list[str]:
    """
    Select top N features by mean absolute SHAP value.

    Fits a lightweight XGBoost model, computes SHAP values, and returns
    the top_n features ranked by mean |SHAP|. Result is cached to disk
    so subsequent runs skip the computation.

    Args:
        X_train:    Training feature matrix.
        y_train:    Training target.
        top_n:      Number of features to keep.
        cache_path: JSON file to cache selected features. Loads if exists.

    Returns:
        List of selected feature names (length = top_n or fewer).
    """
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            features = json.load(f)
        print(f"SHAP features loaded from cache ({len(features)} features)")
        return features

    # Lightweight model just for SHAP ranking
    scout = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )
    scout.fit(X_train, y_train)

    explainer   = shap.TreeExplainer(scout)
    shap_values = explainer.shap_values(X_train)

    # Normalise to 2D (n_samples, n_features):
    #   - Binary      : already 2D
    #   - Multi-class : SHAP ≥0.42 returns 3D (n_samples, n_features, n_classes)
    #                   older versions return a list of 2D arrays
    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        shap_values = np.abs(shap_values).mean(axis=2)  # → (n_samples, n_features)

    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_train.columns,
    ).sort_values(ascending=False)

    features = mean_shap.head(top_n).index.tolist()
    print(f"Top {len(features)} SHAP features selected")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(features, f)

    return features


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    objective: str = "binary",
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Tune XGBoost hyperparameters with Optuna.

    Stage 1 (objective='binary'):
        CV: TimeSeriesSplit — validation is always later than training.
        Metric: PR-AUC (average precision).

    Stage 2 (objective='multi'):
        CV: StratifiedKFold — ensures each fold has all 3 tier classes.
        Metric: weighted F1.

    Args:
        X_train:      Training feature matrix.
        y_train:      Training target.
        objective:    'binary' for Stage 1, 'multi' for Stage 2.
        n_trials:     Number of Optuna trials.
        cv_folds:     Number of CV folds.
        random_state: Random seed.

    Returns:
        Dict of best hyperparameters.
    """
    is_multi = (objective == "multi")

    if is_multi:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        splits = list(cv.split(X_train, y_train))
    else:
        cv = TimeSeriesSplit(n_splits=cv_folds)
        splits = list(cv.split(X_train))

    def objective_fn(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state":     random_state,
            "verbosity":        0,
        }

        scores = []
        for train_idx, val_idx in splits:
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            if is_multi:
                model = xgb.XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    **params,
                )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(f1_score(y_val, preds, average="weighted"))
            else:
                pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    scale_pos_weight=pos_weight,
                    **params,
                )
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_val)[:, 1]
                scores.append(average_precision_score(y_val, probs))

        return np.mean(scores)

    # Seed the TPE sampler so hyperparameter search is reproducible across runs
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)

    print(f"Best score ({objective}): {study.best_value:.4f}")
    return study.best_params


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    objective: str = "binary",
) -> xgb.XGBClassifier:
    """
    Train the final XGBoost model with tuned hyperparameters.

    Args:
        X_train:     Training feature matrix.
        y_train:     Training target.
        best_params: Hyperparameters from tune_hyperparameters().
        objective:   'binary' for Stage 1, 'multi' for Stage 2.

    Returns:
        Fitted XGBoost classifier.
    """
    if objective == "multi":
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            **best_params,
        )
    else:
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=pos_weight,
            **best_params,
        )

    model.fit(X_train, y_train)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def tune_threshold(y_true: pd.Series, probs: np.ndarray, beta: float = 2.0) -> float:
    """
    Find the probability threshold that maximises F-beta score.

    Beta > 1 weights recall higher than precision — appropriate for Stage 1
    where missing a returning customer (false negative) is more costly than
    a false alarm.

    Args:
        y_true: True binary labels.
        probs:  Predicted probabilities for the positive class.
        beta:   F-beta weight (default 2 = recall twice as important).

    Returns:
        Optimal threshold float.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f_beta = (
        (1 + beta**2) * precision * recall
        / (beta**2 * precision + recall + 1e-9)
    )
    return float(thresholds[np.argmax(f_beta)])


def evaluate_stage1(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float | None = None,
) -> dict:
    """
    Evaluate Stage 1 (return classifier) on the held-out test set.

    Metrics:
        pr_auc     : Primary metric — area under precision-recall curve.
        roc_auc    : Area under ROC curve.
        precision  : At the F2-optimal threshold.
        recall     : At the F2-optimal threshold.
        threshold  : Decision threshold used.

    Args:
        model:     Fitted Stage 1 classifier.
        X_test:    Test feature matrix.
        y_test:    True binary labels.
        threshold: If None, computes F2-optimal threshold from test set.

    Returns:
        Dict of metric name → value.
    """
    probs = model.predict_proba(X_test)[:, 1]

    if threshold is None:
        threshold = tune_threshold(y_test, probs, beta=2.0)

    preds = (probs >= threshold).astype(int)

    return {
        "pr_auc":    round(average_precision_score(y_test, probs), 4),
        "roc_auc":   round(roc_auc_score(y_test, probs), 4),
        "precision": round(float(np.where(preds == 1, y_test == preds, 0).sum() / max(preds.sum(), 1)), 4),
        "recall":    round(float((preds[y_test == 1] == 1).mean()), 4),
        "threshold": round(threshold, 4),
    }


def evaluate_stage2(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate Stage 2 (spend-tier classifier) on held-out returners.

    Metrics:
        weighted_f1      : Weighted F1 across all three tiers.
        per_class_f1     : F1 per tier (Low / Mid / High).

    Args:
        model:  Fitted Stage 2 classifier.
        X_test: Test feature matrix (returners only).
        y_test: True tier labels (0=Low, 1=Mid, 2=High).

    Returns:
        Dict of metric name → value.
    """
    preds = model.predict(X_test)
    per_class = f1_score(y_test, preds, average=None, labels=[0, 1, 2])

    return {
        "weighted_f1":  round(f1_score(y_test, preds, average="weighted"), 4),
        "f1_low":       round(per_class[0], 4),
        "f1_mid":       round(per_class[1], 4),
        "f1_high":      round(per_class[2], 4),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(
    stage1_model: xgb.XGBClassifier,
    stage2_model: xgb.XGBClassifier,
    stage1_features: list[str],
    stage2_features: list[str],
    threshold: float,
    path: str,
) -> None:
    """
    Persist both stage models and metadata to a single pickle file.

    Args:
        stage1_model:    Fitted Stage 1 classifier.
        stage2_model:    Fitted Stage 2 classifier.
        stage1_features: Feature names expected by Stage 1.
        stage2_features: Feature names expected by Stage 2.
        threshold:       F2-optimal decision threshold for Stage 1.
        path:            Output file path (.pkl).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "stage1_model":    stage1_model,
        "stage2_model":    stage2_model,
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "threshold":       threshold,
        "tier_labels":     TIER_LABELS,
    }
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Model saved → {path}")


def load_model(path: str) -> dict:
    """
    Load model artifact from disk.

    Returns:
        Dict with stage1_model, stage2_model, stage1_features,
        stage2_features, threshold, tier_labels.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
