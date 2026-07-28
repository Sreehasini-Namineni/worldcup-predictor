import { useEffect, useState } from 'react'
import { api } from './api'
import Scoreboard from './components/Scoreboard'
import PredictionPanel from './components/PredictionPanel'
import MetricsPanel from './components/MetricsPanel'
import SimulationPanel from './components/SimulationPanel'
import RecentPredictions from './components/RecentPredictions'

export default function App() {
  const [teams, setTeams] = useState([])
  const [modelReady, setModelReady] = useState(null)
  const [apiError, setApiError] = useState(null)

  const [prediction, setPrediction] = useState(null)
  const [predictLoading, setPredictLoading] = useState(false)

  const [simResult, setSimResult] = useState(null)
  const [simLoading, setSimLoading] = useState(false)

  const [recent, setRecent] = useState([])
  const [retraining, setRetraining] = useState(false)

  useEffect(() => {
    api.health()
      .then((h) => setModelReady(h.model_ready))
      .catch(() => setApiError('Cannot reach the API. Is the backend running?'))

    api.teams()
      .then(setTeams)
      .catch(() => {})
  }, [])

  const handlePredict = async (payload) => {
    setPredictLoading(true)
    setApiError(null)
    try {
      const result = await api.predict(payload)
      setPrediction(result)
      const feed = await api.recentPredictions(10)
      setRecent(feed)
    } catch (err) {
      setApiError(err.message)
    } finally {
      setPredictLoading(false)
    }
  }

  const handleSimulate = async (payload) => {
    setSimLoading(true)
    setApiError(null)
    try {
      const result = await api.simulate(payload)
      setSimResult(result)
    } catch (err) {
      setApiError(err.message)
    } finally {
      setSimLoading(false)
    }
  }

  const handleRetrain = async () => {
    setRetraining(true)
    setApiError(null)
    try {
      const status = await api.retrain()
      setModelReady(true)
      alert(`Retrain complete in ${status.elapsed_seconds}s — best val accuracy ${(status.best_val_accuracy * 100).toFixed(1)}%`)
    } catch (err) {
      setApiError(err.message)
    } finally {
      setRetraining(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <div className="eyebrow">Match Predictor</div>
          <h1>World Cup Analytics Dashboard</h1>
        </div>
        <div className="header-actions">
          <span className={`status-dot ${modelReady ? 'status-ok' : 'status-off'}`} />
          <span className="mono status-text">
            {modelReady === null ? 'checking…' : modelReady ? 'model ready' : 'model untrained'}
          </span>
          <button className="ghost" onClick={handleRetrain} disabled={retraining}>
            {retraining ? 'Retraining…' : 'Retrain model'}
          </button>
        </div>
      </header>

      {apiError && <div className="banner">{apiError}</div>}

      <Scoreboard
        prediction={prediction}
        teamA={prediction?.team_a}
        teamB={prediction?.team_b}
      />

      <main className="grid">
        <div className="col">
          <PredictionPanel teams={teams} onPredict={handlePredict} loading={predictLoading} />
          <MetricsPanel prediction={prediction} />
        </div>
        <div className="col">
          <SimulationPanel teams={teams} onSimulate={handleSimulate} loading={simLoading} result={simResult} />
        </div>
        <div className="col">
          <RecentPredictions items={recent} />
        </div>
      </main>

      <style>{`
        .app {
          max-width: 1240px;
          margin: 0 auto;
          padding: clamp(1rem, 3vw, 2.5rem);
        }
        .app-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          margin-bottom: 1.5rem;
          flex-wrap: wrap;
          gap: 1rem;
        }
        h1 {
          font-family: var(--font-body);
          font-weight: 700;
          font-size: clamp(1.3rem, 3vw, 1.9rem);
          margin: 0.2rem 0 0 0;
          color: var(--chalk);
        }
        .header-actions {
          display: flex;
          align-items: center;
          gap: 0.6rem;
        }
        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }
        .status-ok { background: var(--win-a); }
        .status-off { background: var(--danger); }
        .status-text {
          font-size: 0.75rem;
          color: var(--chalk-dim);
          margin-right: 0.5rem;
        }
        .banner {
          background: rgba(224, 101, 79, 0.12);
          border: 1px solid var(--danger);
          color: var(--chalk);
          padding: 0.75rem 1rem;
          border-radius: var(--radius);
          font-size: 0.85rem;
          margin-bottom: 1.5rem;
        }
        .grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1.25rem;
          margin-top: 1.5rem;
          align-items: start;
        }
        .col {
          display: flex;
          flex-direction: column;
          gap: 1.25rem;
        }
        @media (max-width: 900px) {
          .grid { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  )
}
