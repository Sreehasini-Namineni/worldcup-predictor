"""
Generates a realistic synthetic dataset of 40,000+ historical international
matches with Elo ratings, FIFA rankings, and match metadata.

In a production version of this project this would be replaced by real
scraped/downloaded data (e.g. eloratings.net, FIFA's published rankings,
and a Kaggle "international football results" dump). Since this environment
has no internet access to those sources, this script generates data with the
same schema and statistically plausible structure, so the rest of the
pipeline (feature engineering -> training -> calibration -> serving) is
fully runnable end-to-end.

Run:
    python -m app.generate_sample_data
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_TEAMS = 60
N_MATCHES = 42000
START_ELO_MEAN = 1500
START_ELO_STD = 120

TEAMS = [
    "Brazil", "Germany", "Argentina", "France", "Italy", "Spain", "England",
    "Netherlands", "Portugal", "Uruguay", "Belgium", "Croatia", "Mexico",
    "USA", "Colombia", "Chile", "Japan", "South Korea", "Morocco", "Senegal",
    "Nigeria", "Ghana", "Cameroon", "Australia", "Switzerland", "Sweden",
    "Denmark", "Poland", "Wales", "Serbia", "Ukraine", "Russia", "Turkey",
    "Egypt", "Tunisia", "Algeria", "Ivory Coast", "Costa Rica", "Ecuador",
    "Peru", "Paraguay", "Venezuela", "Canada", "Jamaica", "Iran", "Saudi Arabia",
    "Qatar", "Iraq", "China", "Norway", "Austria", "Czechia", "Scotland",
    "Ireland", "Greece", "Romania", "Hungary", "Slovakia", "Finland", "Iceland",
]
assert len(TEAMS) == N_TEAMS

CONFEDERATIONS = {
    # rough real-world groupings, used only as a categorical feature
    "UEFA": {"Germany", "France", "Italy", "Spain", "England", "Netherlands",
              "Portugal", "Belgium", "Croatia", "Switzerland", "Sweden",
              "Denmark", "Poland", "Wales", "Serbia", "Ukraine", "Russia",
              "Turkey", "Norway", "Austria", "Czechia", "Scotland", "Ireland",
              "Greece", "Romania", "Hungary", "Slovakia", "Finland", "Iceland"},
    "CONMEBOL": {"Brazil", "Argentina", "Uruguay", "Colombia", "Chile",
                  "Ecuador", "Peru", "Paraguay", "Venezuela"},
    "CONCACAF": {"Mexico", "USA", "Costa Rica", "Canada", "Jamaica"},
    "AFC": {"Japan", "South Korea", "Iran", "Saudi Arabia", "Qatar", "Iraq",
             "China", "Australia"},
    "CAF": {"Morocco", "Senegal", "Nigeria", "Ghana", "Cameroon", "Egypt",
             "Tunisia", "Algeria", "Ivory Coast"},
}


def team_confederation(team: str) -> str:
    for conf, members in CONFEDERATIONS.items():
        if team in members:
            return conf
    return "UEFA"


def simulate_dataset(n_matches: int = N_MATCHES, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    elo = {t: rng.normal(START_ELO_MEAN, START_ELO_STD) for t in TEAMS}
    fifa_rank = {t: r + 1 for r, t in enumerate(rng.permutation(TEAMS))}
    last_match_day = {t: 0 for t in TEAMS}
    recent_results = {t: [] for t in TEAMS}  # list of +1 win / 0 draw / -1 loss
    goals_for_hist = {t: [] for t in TEAMS}
    goals_against_hist = {t: [] for t in TEAMS}
    h2h = {}  # (a, b) -> [wins_a, draws, wins_b]

    rows = []
    day = 0

    for i in range(n_matches):
        day += rng.integers(1, 5)
        a, b = rng.choice(TEAMS, size=2, replace=False)

        elo_diff = elo[a] - elo[b]
        # Standard Elo expected score formula
        expected_a = 1 / (1 + 10 ** (-elo_diff / 400))

        neutral = int(rng.random() < 0.35)
        is_knockout = int(rng.random() < 0.25)

        # Home advantage bump when not neutral (small, since WC group games
        # are mostly neutral-venue anyway).
        home_bump = 0.0 if neutral else 0.03
        p_a = np.clip(expected_a + home_bump, 0.03, 0.97)

        # Draw probability shrinks the more lopsided the match is, and is
        # further reduced in knockout matches (extra time / penalties still
        # produce a "win" in this simplified label space).
        draw_base = 0.28 - 0.15 * abs(p_a - 0.5)
        if is_knockout:
            draw_base *= 0.4
        p_draw = max(draw_base, 0.05)
        p_b = max(1 - p_a - p_draw, 0.02)
        probs = np.array([p_a, p_draw, p_b])
        probs = probs / probs.sum()

        outcome = rng.choice([0, 1, 2], p=probs)  # 0 = A win, 1 = draw, 2 = B win

        # Goals: Poisson-ish, skewed by the same expected_a signal.
        lam_a = 1.1 + 1.8 * p_a
        lam_b = 1.1 + 1.8 * p_b
        goals_a = rng.poisson(lam_a)
        goals_b = rng.poisson(lam_b)
        # Reconcile goals with sampled outcome label so the two stay consistent.
        if outcome == 0 and goals_a <= goals_b:
            goals_a = goals_b + 1
        elif outcome == 2 and goals_b <= goals_a:
            goals_b = goals_a + 1
        elif outcome == 1:
            goals_b = goals_a

        # Elo update (K=30, standard for international football simulations)
        K = 30
        actual_a = 1.0 if outcome == 0 else (0.5 if outcome == 1 else 0.0)
        elo[a] += K * (actual_a - expected_a)
        elo[b] += K * ((1 - actual_a) - (1 - expected_a))

        form_a = np.mean(recent_results[a][-10:]) if recent_results[a] else 0.0
        form_b = np.mean(recent_results[b][-10:]) if recent_results[b] else 0.0

        key = tuple(sorted((a, b)))
        if key not in h2h:
            h2h[key] = [0, 0, 0]
        h2h_a_wins, h2h_draws, h2h_b_wins = h2h[key]
        h2h_total = h2h_a_wins + h2h_draws + h2h_b_wins
        h2h_win_rate_a = (h2h_a_wins / h2h_total) if h2h_total > 0 else 0.5

        rows.append({
            "match_id": i,
            "day": day,
            "team_a": a,
            "team_b": b,
            "elo_team_a": round(elo[a] - K * (actual_a - expected_a), 1),  # pre-match elo
            "elo_team_b": round(elo[b] - K * ((1 - actual_a) - (1 - expected_a)), 1),
            "fifa_rank_team_a": fifa_rank[a],
            "fifa_rank_team_b": fifa_rank[b],
            "form_a": form_a,
            "form_b": form_b,
            "goals_avg_a": np.mean(goals_for_hist[a][-10:]) if goals_for_hist[a] else 1.3,
            "goals_avg_b": np.mean(goals_for_hist[b][-10:]) if goals_for_hist[b] else 1.3,
            "goals_conceded_avg_a": np.mean(goals_against_hist[a][-10:]) if goals_against_hist[a] else 1.3,
            "goals_conceded_avg_b": np.mean(goals_against_hist[b][-10:]) if goals_against_hist[b] else 1.3,
            "h2h_win_rate_a": h2h_win_rate_a,
            "is_knockout": is_knockout,
            "neutral_venue": neutral,
            "confederation_match": int(team_confederation(a) == team_confederation(b)),
            "rest_days_a": day - last_match_day[a],
            "rest_days_b": day - last_match_day[b],
            "goals_a": int(goals_a),
            "goals_b": int(goals_b),
            "outcome": int(outcome),  # 0=A win, 1=draw, 2=B win
        })

        # update rolling state
        recent_results[a].append(1 if outcome == 0 else (0 if outcome == 1 else -1))
        recent_results[b].append(1 if outcome == 2 else (0 if outcome == 1 else -1))
        goals_for_hist[a].append(goals_a)
        goals_for_hist[b].append(goals_b)
        goals_against_hist[a].append(goals_b)
        goals_against_hist[b].append(goals_a)
        last_match_day[a] = day
        last_match_day[b] = day
        if outcome == 0:
            h2h[key][0 if key[0] == a else 2] += 1
        elif outcome == 1:
            h2h[key][1] += 1
        else:
            h2h[key][2 if key[0] == a else 0] += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate_dataset()
    df.to_csv("data/raw_matches.csv", index=False)
    print(f"Wrote {len(df)} matches to data/raw_matches.csv")
