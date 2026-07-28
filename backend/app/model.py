"""
Flax model definition for World Cup match outcome prediction.

The model takes an engineered feature vector for a single match (from the
perspective of "team A vs team B") and predicts a 3-way outcome distribution:
    [P(team A win), P(draw), P(team B win)]

Architecture: a small MLP with dropout + batchnorm-free residual blocks.
Kept intentionally compact since the feature set is low-dimensional
(~12-16 engineered features) and the dataset is on the order of tens of
thousands of rows, not millions -- a large network would overfit.
"""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

# Feature order used everywhere in the pipeline. Keeping this as a single
# source of truth avoids column-order bugs between training and inference.
FEATURE_COLUMNS: Sequence[str] = [
    "elo_diff",  # team_a_elo - team_b_elo
    "elo_team_a",
    "elo_team_b",
    "fifa_rank_diff",  # team_b_rank - team_a_rank (positive favors A)
    "fifa_rank_team_a",
    "fifa_rank_team_b",
    "form_diff",  # rolling win-rate diff over last 10 matches
    "goals_scored_avg_a",
    "goals_scored_avg_b",
    "goals_conceded_avg_a",
    "goals_conceded_avg_b",
    "h2h_win_rate_a",  # historical head-to-head win rate for team A
    "is_knockout",  # 0/1 -- knockout matches behave differently than groups
    "neutral_venue",  # 0/1
    "confederation_match",  # 0/1 -- same confederation qualifiers, proxy feature
    "rest_days_diff",  # days since last match, A minus B
]

NUM_FEATURES = len(FEATURE_COLUMNS)
NUM_CLASSES = 3  # [win_a, draw, win_b]


class MatchOutcomeMLP(nn.Module):
    """Small feed-forward network with dropout regularization."""

    hidden_sizes: Sequence[int] = (64, 32, 16)
    dropout_rate: float = 0.25

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_sizes):
            residual = x
            x = nn.Dense(size, name=f"dense_{i}")(x)
            x = nn.LayerNorm(name=f"ln_{i}")(x)
            x = nn.gelu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            # Project residual if shapes don't match, otherwise add directly.
            if residual.shape[-1] == size:
                x = x + residual

        logits = nn.Dense(NUM_CLASSES, name="output")(x)
        return logits  # raw logits; softmax + calibration applied downstream


def init_model(rng, learning_rate: float = 1e-3):
    """Convenience initializer returning a fresh model + example params shape."""
    model = MatchOutcomeMLP()
    dummy_input = jnp.ones((1, NUM_FEATURES))
    params = model.init({"params": rng, "dropout": rng}, dummy_input, train=False)
    return model, params
