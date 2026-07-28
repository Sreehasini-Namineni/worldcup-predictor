"""
Monte Carlo tournament simulator. Runs N simulated brackets end-to-end using
the calibrated model's win probabilities at each matchup, and aggregates how
often each team reaches / wins each round -- this is what powers the
dashboard's "10+ tournament simulations" feature.
"""

import math
from collections import defaultdict
from typing import Dict, List

import numpy as np

ROUND_NAMES = ["Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion"]


def simulate_bracket(
    teams: List[str],
    predict_fn,  # callable: (team_a, team_b) -> (p_win_a, p_draw, p_win_b)
    n_simulations: int = 1000,
    seed: int = 0,
) -> Dict:
    """
    predict_fn should return calibrated probabilities for a single matchup.
    Knockout matches have no draw outcome in the final result (extra
    time/penalties resolve it), so we redistribute drawn mass proportionally
    to the two win probabilities -- a standard trick for turning a 3-way
    market into a 2-way knockout market.
    """
    assert math.log2(len(teams)).is_integer(), "Bracket size must be a power of 2"
    rng = np.random.default_rng(seed)

    n_rounds = int(math.log2(len(teams)))
    round_names = ROUND_NAMES[-(n_rounds + 1):-1] if n_rounds < len(ROUND_NAMES) else \
        [f"Round {i+1}" for i in range(n_rounds)]

    champion_counts = defaultdict(int)
    round_reach_counts = [defaultdict(int) for _ in range(n_rounds)]

    # Cache matchup probabilities so we don't call the model N times for the
    # same pairing across simulations.
    prob_cache = {}

    def get_knockout_prob(a, b):
        key = (a, b)
        if key not in prob_cache:
            p_a, p_draw, p_b = predict_fn(a, b)
            # redistribute draw mass proportionally
            total = p_a + p_b
            adj_a = p_a + p_draw * (p_a / total if total > 0 else 0.5)
            prob_cache[key] = min(max(adj_a, 0.01), 0.99)
        return prob_cache[key]

    for _ in range(n_simulations):
        current_round = list(teams)
        for r in range(n_rounds):
            next_round = []
            for i in range(0, len(current_round), 2):
                a, b = current_round[i], current_round[i + 1]
                p_a = get_knockout_prob(a, b)
                winner = a if rng.random() < p_a else b
                next_round.append(winner)
                round_reach_counts[r][winner] += 1
            current_round = next_round
        champion_counts[current_round[0]] += 1

    rounds_output = []
    for r, name in enumerate(round_names):
        probs = {team: count / n_simulations for team, count in round_reach_counts[r].items()}
        rounds_output.append({"round_name": name, "team_win_probabilities": probs})

    champion_probabilities = {team: count / n_simulations for team, count in champion_counts.items()}

    return {
        "n_simulations": n_simulations,
        "champion_probabilities": dict(sorted(champion_probabilities.items(), key=lambda kv: -kv[1])),
        "rounds": rounds_output,
    }
