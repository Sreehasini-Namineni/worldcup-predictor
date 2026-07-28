"""
Post-hoc probability calibration.

The Flax model's raw softmax outputs tend to be overconfident (a known
property of neural net classifiers). We fit a simple temperature-scaling
parameter T on the validation set, minimizing NLL, so downstream expected
value / confidence-interval calculations in the dashboard are trustworthy
rather than just "the model's raw softmax."
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from app.data_pipeline import build_training_arrays
from app.model import MatchOutcomeMLP


def get_logits(params, X):
    model = MatchOutcomeMLP()
    return model.apply({"params": params}, jnp.array(X), train=False, rngs={"dropout": jax.random.PRNGKey(0)})


def fit_temperature(logits: jnp.ndarray, y: jnp.ndarray, steps: int = 500, lr: float = 0.05) -> float:
    """Fits a single scalar T minimizing cross-entropy of logits/T against y."""
    log_t = jnp.array(0.0)  # optimize in log-space so T stays positive
    opt = optax.adam(lr)
    opt_state = opt.init(log_t)

    one_hot = jax.nn.one_hot(y, num_classes=3)

    def loss_fn(log_t):
        t = jnp.exp(log_t)
        scaled = logits / t
        return optax.softmax_cross_entropy(scaled, one_hot).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    for _ in range(steps):
        loss, grad = grad_fn(log_t)
        updates, opt_state = opt.update(grad, opt_state)
        log_t = optax.apply_updates(log_t, updates)

    return float(jnp.exp(log_t))


def calibrated_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    return np.array(jax.nn.softmax(scaled, axis=-1))


def run_calibration(params) -> float:
    arrays = build_training_arrays()
    logits = get_logits(params, arrays["X_val"])
    T = fit_temperature(logits, jnp.array(arrays["y_val"]))
    print(f"Fitted calibration temperature T={T:.3f}")
    np.save("data/model/temperature.npy", np.array([T]))
    return T


if __name__ == "__main__":
    from flax import serialization
    from app.model import init_model
    import jax as _jax

    _, params_shape = init_model(_jax.random.PRNGKey(0))
    with open("data/model/params.msgpack", "rb") as f:
        params = serialization.from_bytes(params_shape["params"], f.read())
    run_calibration(params)
