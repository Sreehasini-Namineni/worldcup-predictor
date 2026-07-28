import { useState } from 'react'

export default function PredictionPanel({ teams, onPredict, loading }) {
  const [teamA, setTeamA] = useState('')
  const [teamB, setTeamB] = useState('')
  const [isKnockout, setIsKnockout] = useState(false)
  const [neutralVenue, setNeutralVenue] = useState(true)

  const canSubmit = teamA && teamB && teamA !== teamB && !loading

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!canSubmit) return
    onPredict({ team_a: teamA, team_b: teamB, is_knockout: isKnockout, neutral_venue: neutralVenue })
  }

  return (
    <form className="card" onSubmit={handleSubmit}>
      <div className="card-title">Set the matchup</div>

      <div className="field-row">
        <label className="field">
          <span className="eyebrow">Team A</span>
          <select value={teamA} onChange={(e) => setTeamA(e.target.value)}>
            <option value="">Select team</option>
            {teams.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="eyebrow">Team B</span>
          <select value={teamB} onChange={(e) => setTeamB(e.target.value)}>
            <option value="">Select team</option>
            {teams.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="toggle-row">
        <label className="toggle">
          <input type="checkbox" checked={isKnockout} onChange={(e) => setIsKnockout(e.target.checked)} />
          Knockout stage
        </label>
        <label className="toggle">
          <input type="checkbox" checked={neutralVenue} onChange={(e) => setNeutralVenue(e.target.checked)} />
          Neutral venue
        </label>
      </div>

      <button type="submit" className="primary" disabled={!canSubmit}>
        {loading ? 'Running inference…' : 'Predict outcome'}
      </button>

      {teamA && teamB && teamA === teamB && (
        <p className="warn mono">Pick two different teams.</p>
      )}

      <style>{`
        .field-row {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1rem;
          margin-bottom: 1rem;
        }
        .field {
          display: flex;
          flex-direction: column;
          gap: 0.4rem;
        }
        select { width: 100%; }
        .toggle-row {
          display: flex;
          gap: 1.5rem;
          margin-bottom: 1.25rem;
        }
        .toggle {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.85rem;
          color: var(--chalk-dim);
          cursor: pointer;
        }
        .toggle input { accent-color: var(--gold); }
        .warn {
          color: var(--danger);
          font-size: 0.78rem;
          margin: 0.75rem 0 0 0;
        }
        @media (max-width: 520px) {
          .field-row { grid-template-columns: 1fr; }
        }
      `}</style>
    </form>
  )
}
