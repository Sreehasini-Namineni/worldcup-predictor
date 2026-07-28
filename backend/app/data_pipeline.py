"""
Preprocessing pipeline: turns raw match rows into the engineered feature
matrix consumed by the Flax model (see model.FEATURE_COLUMNS).

This is deliberately split into pure, cacheable functions so the "retrain
in under 10 minutes" pipeline can skip re-deriving features when the raw
data hasn't changed -- see build_training_arrays(..., use_cache=True).
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.model import FEATURE_COLUMNS

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
CACHE_DIR = DATA_DIR / "cache"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model's feature columns from raw match rows."""
    out = pd.DataFrame(index=df.index)
    out["elo_diff"] = df["elo_team_a"] - df["elo_team_b"]
    out["elo_team_a"] = df["elo_team_a"]
    out["elo_team_b"] = df["elo_team_b"]
    out["fifa_rank_diff"] = df["fifa_rank_team_b"] - df["fifa_rank_team_a"]
    out["fifa_rank_team_a"] = df["fifa_rank_team_a"]
    out["fifa_rank_team_b"] = df["fifa_rank_team_b"]
    out["form_diff"] = df["form_a"] - df["form_b"]
    out["goals_scored_avg_a"] = df["goals_avg_a"]
    out["goals_scored_avg_b"] = df["goals_avg_b"]
    out["goals_conceded_avg_a"] = df["goals_conceded_avg_a"]
    out["goals_conceded_avg_b"] = df["goals_conceded_avg_b"]
    out["h2h_win_rate_a"] = df["h2h_win_rate_a"]
    out["is_knockout"] = df["is_knockout"].astype(float)
    out["neutral_venue"] = df["neutral_venue"].astype(float)
    out["confederation_match"] = df["confederation_match"].astype(float)
    out["rest_days_diff"] = df["rest_days_a"] - df["rest_days_b"]

    assert list(out.columns) == list(FEATURE_COLUMNS), "Feature column drift detected"
    return out


def normalize_features(X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
    """Z-score normalization. Returns (X_norm, mean, std) so mean/std computed
    on train data can be reapplied to validation/inference data."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def _raw_data_hash(csv_path: Path) -> str:
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_training_arrays(csv_path: str = None, use_cache: bool = True):
    """
    Loads raw matches, engineers features, splits train/val/test, and returns
    numpy arrays ready for JAX. Caches the engineered arrays keyed on a hash
    of the raw CSV so repeated retrains (e.g. triggered from the dashboard)
    skip the pandas feature-engineering pass entirely when the underlying
    data hasn't changed -- this is most of what gets total retrain time
    under 10 minutes.
    """
    csv_path = Path(csv_path or (DATA_DIR / "raw_matches.csv"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = _raw_data_hash(csv_path)
    cache_file = CACHE_DIR / f"features_{cache_key}.npz"

    if use_cache and cache_file.exists():
        npz = np.load(cache_file)
        return {k: npz[k] for k in npz.files}

    df = pd.read_csv(csv_path)
    df = df.sort_values("day").reset_index(drop=True)

    features = engineer_features(df)
    X = features.values.astype(np.float32)
    y = df["outcome"].values.astype(np.int32)

    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    X_train, mean, std = normalize_features(X_train)
    X_val, _, _ = normalize_features(X_val, mean, std)
    X_test, _, _ = normalize_features(X_test, mean, std)

    arrays = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_mean": mean, "feature_std": std,
    }
    np.savez(cache_file, **arrays)
    return arrays
