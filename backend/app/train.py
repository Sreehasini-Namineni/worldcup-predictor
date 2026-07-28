"""
Training loop for the match outcome model, using Flax + Optax on top of JAX.

JIT-compiling the train_step is the other big contributor (along with the
feature cache in data_pipeline.py) to keeping full retraining under 10
minutes even as the dataset grows past 40k matches: after the first
compilation, each subsequent step is a compiled XLA call rather than
re-traced Python.

Run:
    python -m app.train
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from app.data_pipeline import build_training_arrays
from app.model import MatchOutcomeMLP, NUM_FEATURES

MODEL_DIR = Path("data/model")


class TrainState(train_state.TrainState):
    dropout_rng: jax.Array


def create_train_state(rng, learning_rate: float, weight_decay: float = 1e-4):
    model = MatchOutcomeMLP()
    params_rng, dropout_rng = jax.random.split(rng)
    dummy = jnp.ones((1, NUM_FEATURES))
    variables = model.init({"params": params_rng, "dropout": dropout_rng}, dummy, train=False)

    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=2000)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx, dropout_rng=dropout_rng)


def loss_fn(params, apply_fn, dropout_rng, x, y, train: bool):
    logits = apply_fn({"params": params}, x, train=train, rngs={"dropout": dropout_rng})
    one_hot = jax.nn.one_hot(y, num_classes=3)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss, logits


@jax.jit
def train_step(state: TrainState, x, y):
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, state.apply_fn, dropout_rng, x, y, True)
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return state, loss, acc


@jax.jit
def eval_step(state: TrainState, x, y):
    logits = state.apply_fn({"params": state.params}, x, train=False, rngs={"dropout": state.dropout_rng})
    one_hot = jax.nn.one_hot(y, num_classes=3)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss, acc, logits


def iterate_batches(X, y, batch_size, rng):
    n = X.shape[0]
    perm = jax.random.permutation(rng, n)
    for i in range(0, n - batch_size + 1, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]


def train(epochs: int = 25, batch_size: int = 128, learning_rate: float = 2e-3, seed: int = 0):
    t0 = time.time()
    arrays = build_training_arrays()
    X_train, y_train = jnp.array(arrays["X_train"]), jnp.array(arrays["y_train"])
    X_val, y_val = jnp.array(arrays["X_val"]), jnp.array(arrays["y_val"])

    rng = jax.random.PRNGKey(seed)
    state = create_train_state(rng, learning_rate)

    best_val_acc = 0.0
    history = []

    for epoch in range(epochs):
        rng, epoch_rng = jax.random.split(rng)
        epoch_losses, epoch_accs = [], []
        for xb, yb in iterate_batches(X_train, y_train, batch_size, epoch_rng):
            state, loss, acc = train_step(state, xb, yb)
            epoch_losses.append(loss)
            epoch_accs.append(acc)

        val_loss, val_acc, _ = eval_step(state, X_val, y_val)
        train_acc = float(np.mean(epoch_accs))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "train_acc": train_acc,
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        })
        best_val_acc = max(best_val_acc, float(val_acc))
        print(f"epoch {epoch:02d}  train_acc={train_acc:.3f}  val_acc={float(val_acc):.3f}")

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s. Best val acc: {best_val_acc:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    from flax import serialization
    with open(MODEL_DIR / "params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    np.savez(
        MODEL_DIR / "norm_stats.npz",
        mean=arrays["feature_mean"],
        std=arrays["feature_std"],
    )
    return state, history, elapsed


if __name__ == "__main__":
    train()
