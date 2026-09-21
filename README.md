# World Cup Match Predictor

A machine learning system that predicts World Cup match outcomes, with an
automated retraining pipeline and an interactive analytics dashboard for
match predictions, expected value, confidence metrics, and Monte Carlo
tournament simulations.

**Stack:** JAX · Flax · Optax · FastAPI · PostgreSQL · React · Docker

---

## Architecture

```
worldcup-predictor/
├── backend/
│   ├── app/
│   │   ├── model.py              # Flax MLP architecture + feature schema
│   │   ├── generate_sample_data.py  # synthetic historical match generator
│   │   ├── data_pipeline.py      # feature engineering + caching
│   │   ├── train.py              # Optax training loop, JIT-compiled
│   │   ├── calibrate.py          # temperature scaling for calibrated probs
│   │   ├── simulate.py           # Monte Carlo tournament bracket engine
│   │   ├── database.py           # SQLAlchemy engine/session
│   │   ├── models_db.py          # ORM models (teams, matches, predictions, sims)
│   │   ├── schemas.py            # Pydantic request/response models
│   │   └── main.py               # FastAPI app + endpoints
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Scoreboard.jsx        # hero: live win/draw/win odds
│   │   │   ├── PredictionPanel.jsx   # team selection form
│   │   │   ├── MetricsPanel.jsx      # expected value breakdown
│   │   │   ├── SimulationPanel.jsx   # bracket builder + championship chart
│   │   │   └── RecentPredictions.jsx # session prediction feed
│   │   ├── api.js
│   │   ├── App.jsx
│   │   └── index.css             # design tokens (stadium scoreboard theme)
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```

### How the pieces fit together

1. **`generate_sample_data.py`** produces a synthetic dataset of 40,000+
   historical matches with plausible Elo trajectories, FIFA rankings, goal
   history, head-to-head records, and rest days. *(This stands in for
   scraped data from eloratings.net / FIFA / Kaggle — see "Using real
   data" below.)*
2. **`data_pipeline.py`** engineers 16 features per match (Elo diff, rank
   diff, rolling form, goals for/against, head-to-head win rate, rest-day
   diff, knockout/neutral-venue flags) and caches the resulting arrays keyed
   on a hash of the raw CSV, so retrains skip re-deriving features when the
   underlying data hasn't changed.
3. **`train.py`** trains a small residual MLP in Flax using Optax (AdamW,
   cosine decay, gradient clipping). The train/eval steps are JIT-compiled,
   which combined with the feature cache is what keeps full retrains under
   10 minutes even as the dataset grows — in testing, 25 epochs over ~42k
   matches completed in well under a minute on CPU.
4. **`calibrate.py`** fits a temperature-scaling parameter on the
   validation set so the softmax outputs aren't overconfident, which
   matters for the EV and confidence numbers shown in the dashboard to be
   meaningful rather than just raw (miscalibrated) softmax scores.
5. **`main.py`** (FastAPI) loads the trained params + calibration
   temperature into memory once, then serves `/predict` and `/simulate`
   with fast synchronous inference, logging every prediction to Postgres.
6. **`simulate.py`** runs N simulated single-elimination brackets, at each
   matchup drawing a winner from the model's calibrated win probability
   (draw probability is redistributed proportionally, since knockout
   matches always produce a winner), and aggregates championship / round
   odds across simulations.
7. The **React dashboard** calls these endpoints, rendering live
   predictions in a scoreboard-style hero, an EV table, a bracket builder
   with a championship-odds chart (Recharts), and a feed of recent
   predictions pulled from Postgres.
