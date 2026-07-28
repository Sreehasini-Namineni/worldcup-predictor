import time
from pathlib import Path
from typing import Optional

import jax
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.calibrate import calibrated_probs
from app.data_pipeline import engineer_features
from app.database import Base, engine, get_db
from app.model import FEATURE_COLUMNS, MatchOutcomeMLP, init_model
from app.models_db import PredictionLog, SimulationRun
from app.schemas import (MatchInput, PredictionResponse, SimulationRequest,
                          SimulationResponse, TrainingStatus)
from app.simulate import simulate_bracket

MODEL_DIR = Path("data/model")

app = FastAPI(title="World Cup Match Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)


class ModelServer:
    """Holds the loaded Flax params + calibration temperature + team lookup
    tables in memory, and exposes a synchronous predict() used by both the
    REST endpoints and the tournament simulator."""

    def __init__(self):
        self.model = MatchOutcomeMLP()
        self.params = None
        self.temperature = 1.0
        self.feature_mean = None
        self.feature_std = None
        self.team_lookup = {}
        self._load()

    def _load(self):
        try:
            from flax import serialization
            _, params_shape = init_model(jax.random.PRNGKey(0))
            with open(MODEL_DIR / "params.msgpack", "rb") as f:
                self.params = serialization.from_bytes(params_shape["params"], f.read())

            norm = np.load(MODEL_DIR / "norm_stats.npz")
            self.feature_mean, self.feature_std = norm["mean"], norm["std"]

            temp_path = MODEL_DIR / "temperature.npy"
            if temp_path.exists():
                self.temperature = float(np.load(temp_path)[0])
        except FileNotFoundError:
            # Model hasn't been trained yet -- endpoints will report 503 until
            # POST /train is called.
            self.params = None

        self._load_team_lookup()

    def _load_team_lookup(self):
        import pandas as pd
        raw_path = Path("data/raw_matches.csv")
        if not raw_path.exists():
            return
        df = pd.read_csv(raw_path)
        latest = {}
        for _, row in df.sort_values("day").iterrows():
            latest[row["team_a"]] = {"elo": row["elo_team_a"], "fifa_rank": row["fifa_rank_team_a"]}
            latest[row["team_b"]] = {"elo": row["elo_team_b"], "fifa_rank": row["fifa_rank_team_b"]}
        self.team_lookup = latest

    @property
    def is_ready(self) -> bool:
        return self.params is not None

    def build_feature_row(self, team_a: str, team_b: str, is_knockout: bool, neutral_venue: bool):
        a = self.team_lookup.get(team_a, {"elo": 1500, "fifa_rank": 50})
        b = self.team_lookup.get(team_b, {"elo": 1500, "fifa_rank": 50})
        row = {
            "elo_team_a": a["elo"], "elo_team_b": b["elo"],
            "fifa_rank_team_a": a["fifa_rank"], "fifa_rank_team_b": b["fifa_rank"],
            "form_a": 0.0, "form_b": 0.0,
            "goals_avg_a": 1.4, "goals_avg_b": 1.4,
            "goals_conceded_avg_a": 1.2, "goals_conceded_avg_b": 1.2,
            "h2h_win_rate_a": 0.5,
            "is_knockout": int(is_knockout), "neutral_venue": int(neutral_venue),
            "confederation_match": 0,
            "rest_days_a": 7, "rest_days_b": 7,
        }
        import pandas as pd
        df = pd.DataFrame([row])
        features = engineer_features(df)
        return features.values.astype(np.float32)

    def predict(self, team_a: str, team_b: str, is_knockout: bool = False, neutral_venue: bool = True):
        if not self.is_ready:
            raise HTTPException(503, "Model not trained yet. POST /train first.")
        X = self.build_feature_row(team_a, team_b, is_knockout, neutral_venue)
        X_norm = (X - self.feature_mean) / self.feature_std
        logits = self.model.apply(
            {"params": self.params}, X_norm, train=False, rngs={"dropout": jax.random.PRNGKey(0)}
        )
        probs = calibrated_probs(np.array(logits), self.temperature)[0]
        return float(probs[0]), float(probs[1]), float(probs[2])


model_server = ModelServer()


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model_server.is_ready}


@app.get("/teams")
def list_teams():
    return sorted(model_server.team_lookup.keys())


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: MatchInput, db: Session = Depends(get_db)):
    p_a, p_draw, p_b = model_server.predict(
        payload.team_a, payload.team_b, payload.is_knockout, payload.neutral_venue
    )

    # Confidence: 1 - normalized Shannon entropy (0 = totally uncertain, 1 = certain)
    probs = np.array([p_a, p_draw, p_b])
    probs = np.clip(probs, 1e-9, 1)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(3)
    confidence = float(1 - entropy / max_entropy)

    # Expected value per $1 stake against "fair" odds implied by our own
    # probabilities (i.e. what a bettor's edge would be if the market priced
    # differently -- here we surface EV vs. a naive 33/33/33 market as a
    # simple illustrative baseline).
    naive_prob = 1 / 3
    expected_value = {
        "win_a": round((p_a / naive_prob) - 1, 4),
        "draw": round((p_draw / naive_prob) - 1, 4),
        "win_b": round((p_b / naive_prob) - 1, 4),
    }

    log = PredictionLog(
        team_a=payload.team_a, team_b=payload.team_b,
        prob_win_a=p_a, prob_draw=p_draw, prob_win_b=p_b,
        confidence=confidence,
    )
    db.add(log)
    db.commit()

    return PredictionResponse(
        team_a=payload.team_a, team_b=payload.team_b,
        prob_win_a=round(p_a, 4), prob_draw=round(p_draw, 4), prob_win_b=round(p_b, 4),
        confidence=round(confidence, 4),
        expected_value=expected_value,
    )


@app.post("/simulate", response_model=SimulationResponse)
def simulate(payload: SimulationRequest, db: Session = Depends(get_db)):
    def predict_fn(a, b):
        return model_server.predict(a, b, is_knockout=True, neutral_venue=True)

    result = simulate_bracket(payload.teams, predict_fn, payload.n_simulations)

    run = SimulationRun(
        n_simulations=payload.n_simulations,
        bracket={"teams": payload.teams},
        results=result,
    )
    db.add(run)
    db.commit()

    return result


@app.post("/train", response_model=TrainingStatus)
def retrain():
    """Kicks off a full retrain: feature caching -> Optax training loop ->
    temperature calibration -> reload into the live ModelServer.
    Synchronous for simplicity; in production this would be a background
    task / job queue so the endpoint returns immediately."""
    from app.train import train as run_training
    from app.calibrate import run_calibration

    t0 = time.time()
    state, history, elapsed = run_training(epochs=25)
    run_calibration(state.params)
    model_server._load()

    return TrainingStatus(
        status="complete",
        elapsed_seconds=round(time.time() - t0, 2),
        best_val_accuracy=max(h["val_acc"] for h in history),
        history=history,
    )


@app.get("/predictions/recent")
def recent_predictions(limit: int = 20, db: Session = Depends(get_db)):
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "team_a": r.team_a, "team_b": r.team_b,
            "prob_win_a": r.prob_win_a, "prob_draw": r.prob_draw, "prob_win_b": r.prob_win_b,
            "confidence": r.confidence, "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
