from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                         Boolean, JSON)
from sqlalchemy.orm import relationship

from app.database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True, nullable=False)
    confederation = Column(String, nullable=True)
    current_elo = Column(Float, nullable=False, default=1500.0)
    fifa_rank = Column(Integer, nullable=True)


class Match(Base):
    """Historical match rows used for training / backtesting."""
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    day = Column(Integer, index=True)
    team_a = Column(String, index=True)
    team_b = Column(String, index=True)
    goals_a = Column(Integer)
    goals_b = Column(Integer)
    outcome = Column(Integer)  # 0 = A win, 1 = draw, 2 = B win
    is_knockout = Column(Boolean, default=False)
    neutral_venue = Column(Boolean, default=False)


class PredictionLog(Base):
    """Every prediction served through the API, for auditability + the
    dashboard's 'recent predictions' feed."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    team_a = Column(String, nullable=False)
    team_b = Column(String, nullable=False)
    prob_win_a = Column(Float, nullable=False)
    prob_draw = Column(Float, nullable=False)
    prob_win_b = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)  # e.g. 1 - entropy-normalized
    model_version = Column(String, default="v1")


class SimulationRun(Base):
    """Stores a full tournament Monte Carlo simulation so the dashboard can
    render results without recomputing on every page load."""
    __tablename__ = "simulation_runs"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    n_simulations = Column(Integer, nullable=False)
    bracket = Column(JSON, nullable=False)  # input bracket structure
    results = Column(JSON, nullable=False)  # team -> win probability by round
