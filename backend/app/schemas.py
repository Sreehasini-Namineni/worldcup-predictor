from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MatchInput(BaseModel):
    team_a: str
    team_b: str
    is_knockout: bool = False
    neutral_venue: bool = True


class PredictionResponse(BaseModel):
    team_a: str
    team_b: str
    prob_win_a: float
    prob_draw: float
    prob_win_b: float
    confidence: float = Field(..., description="1 - normalized entropy of the outcome distribution")
    expected_value: Dict[str, float] = Field(
        ..., description="EV per $1 stake at fair implied odds for each outcome"
    )
    model_version: str = "v1"


class TeamStats(BaseModel):
    name: str
    elo: float
    fifa_rank: int
    confederation: Optional[str] = None


class SimulationRequest(BaseModel):
    teams: List[str] = Field(..., min_length=4, description="Bracket teams, power-of-two length")
    n_simulations: int = Field(1000, ge=10, le=20000)


class RoundResult(BaseModel):
    round_name: str
    team_win_probabilities: Dict[str, float]


class SimulationResponse(BaseModel):
    n_simulations: int
    champion_probabilities: Dict[str, float]
    rounds: List[RoundResult]


class TrainingStatus(BaseModel):
    status: str
    elapsed_seconds: Optional[float] = None
    best_val_accuracy: Optional[float] = None
    history: Optional[list] = None
