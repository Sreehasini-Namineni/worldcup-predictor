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

---

## Running it

### Option A — Docker Compose (recommended)

```bash
docker compose up --build
```

This will, on first run:
1. Start Postgres and wait for it to be healthy.
2. Generate the synthetic 42k-match dataset if it doesn't already exist.
3. Train the model and fit calibration if no trained params are cached.
4. Start the FastAPI server on **http://localhost:8000**.
5. Build and serve the React dashboard on **http://localhost:5173**.

First boot takes a few minutes (image builds + first training run); after
that, restarts are fast since the trained model and dataset persist in
`backend/data/` (mounted as a volume).

To force a full retrain later, either delete `backend/data/model/` and
restart, or use the **"Retrain model"** button in the dashboard header,
which calls `POST /train` directly.

### Option B — Run locally without Docker

**Backend:**
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start Postgres yourself (or point DATABASE_URL at any Postgres instance)
export DATABASE_URL=postgresql://wc_user:wc_password@localhost:5432/worldcup

python -m app.generate_sample_data   # writes backend/data/raw_matches.csv
python -m app.train                  # trains + saves params to data/model/
python -m app.calibrate              # fits calibration temperature

uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
cp .env.example .env    # VITE_API_BASE_URL=/api, proxied to localhost:8000 in dev
npm install
npm run dev
```

Visit **http://localhost:5173**.

---

## API reference

| Method | Path                  | Description                                          |
|--------|------------------------|-------------------------------------------------------|
| GET    | `/health`              | Model load status                                     |
| GET    | `/teams`                | List of teams known to the model (with latest Elo/rank) |
| POST   | `/predict`              | `{team_a, team_b, is_knockout, neutral_venue}` → win/draw/win probabilities, confidence, EV |
| POST   | `/simulate`             | `{teams: [...8], n_simulations}` → championship odds + per-round survival odds |
| POST   | `/train`                | Triggers a full retrain (data → features → train → calibrate → reload) |
| GET    | `/predictions/recent`   | Last N predictions logged to Postgres                 |

Interactive docs are available at `http://localhost:8000/docs` once the
backend is running (FastAPI's built-in Swagger UI).

---

## Using real data instead of synthetic data

`generate_sample_data.py` exists because this environment can't reach
eloratings.net, FIFA's ranking archive, or Kaggle. To swap in real data:

1. Source historical match results (e.g. a "World Cup results 1930–2026"
   CSV), historical Elo ratings, and historical FIFA rankings.
2. Produce a CSV matching the schema in `simulate_dataset()`'s output
   (`team_a`, `team_b`, `elo_team_a`, `elo_team_b`, `fifa_rank_team_a`,
   `fifa_rank_team_b`, `goals_a`, `goals_b`, `outcome`, `is_knockout`,
   `neutral_venue`, `day`, etc.) — or adjust `data_pipeline.engineer_features`
   to match whatever columns your source data has.
3. Save it to `backend/data/raw_matches.csv` and run `python -m app.train`.

The rest of the pipeline (feature engineering, caching, training,
calibration, serving) works unchanged.

---

## Design notes

The dashboard's visual direction is a floodlit night-pitch scoreboard: deep
turf-green background with a subtle mown-stripe texture, chalk-white
touchline text, and a single trophy-gold accent reserved for the model's
live call. Scoreboard digits use a heavy grotesque display face; odds and
stats use a monospace face, evoking a stadium clock or betting board.
