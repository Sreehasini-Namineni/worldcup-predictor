import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const BRACKET_SIZE = 8
const GOLD_SHADES = ['#e3b341', '#cfa23b', '#bb9036', '#a67e30', '#926d2b', '#7d5b25', '#684920', '#54381a']

export default function SimulationPanel({ teams, onSimulate, loading, result }) {
  const [selected, setSelected] = useState([])

  const toggleTeam = (team) => {
    setSelected((prev) => {
      if (prev.includes(team)) return prev.filter((t) => t !== team)
      if (prev.length >= BRACKET_SIZE) return prev
      return [...prev, team]
    })
  }

  const runSimulation = () => {
    if (selected.length !== BRACKET_SIZE) return
    onSimulate({ teams: selected, n_simulations: 2000 })
  }

  const chartData = result
    ? Object.entries(result.champion_probabilities)
        .slice(0, 8)
        .map(([team, prob]) => ({ team, prob: Math.round(prob * 1000) / 10 }))
    : []

  return (
    <div className="card">
      <div className="card-title">Tournament simulation</div>
      <p className="subtext">
        Pick exactly {BRACKET_SIZE} teams for a knockout bracket. Each run plays {' '}
        <strong>2,000</strong> simulated brackets through the model to estimate title odds.
      </p>

      <div className="team-grid">
        {teams.map((t) => (
          <button
            key={t}
            type="button"
            className={`chip ${selected.includes(t) ? 'chip-active' : ''}`}
            onClick={() => toggleTeam(t)}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="sim-footer">
        <span className="eyebrow">{selected.length} / {BRACKET_SIZE} selected</span>
        <button
          type="button"
          className="primary"
          disabled={selected.length !== BRACKET_SIZE || loading}
          onClick={runSimulation}
        >
          {loading ? 'Simulating…' : 'Run simulation'}
        </button>
      </div>

      {chartData.length > 0 && (
        <div className="chart-wrap">
          <div className="eyebrow" style={{ marginBottom: '0.5rem' }}>Championship probability</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 24 }}>
              <XAxis type="number" domain={[0, 'dataMax']} tick={{ fill: 'var(--chalk-dim)', fontSize: 11 }} unit="%" />
              <YAxis type="category" dataKey="team" width={90} tick={{ fill: 'var(--chalk)', fontSize: 12 }} />
              <Tooltip
                contentStyle={{ background: 'var(--pitch-mid)', border: '1px solid rgba(242,240,228,0.15)', borderRadius: 4 }}
                labelStyle={{ color: 'var(--chalk)' }}
                formatter={(v) => [`${v}%`, 'Title chance']}
              />
              <Bar dataKey="prob" radius={[0, 3, 3, 0]}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={GOLD_SHADES[i % GOLD_SHADES.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <style>{`
        .subtext {
          font-size: 0.8rem;
          color: var(--chalk-dim);
          margin: -0.5rem 0 1.25rem 0;
          line-height: 1.5;
        }
        .team-grid {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-bottom: 1.25rem;
          max-height: 200px;
          overflow-y: auto;
          padding-right: 0.25rem;
        }
        .chip {
          background: var(--pitch-mid);
          border: 1px solid rgba(242, 240, 228, 0.14);
          color: var(--chalk-dim);
          font-family: var(--font-mono);
          font-size: 0.75rem;
          padding: 0.4rem 0.7rem;
          border-radius: 999px;
          cursor: pointer;
        }
        .chip:hover { border-color: var(--gold-dim); }
        .chip-active {
          background: var(--gold);
          border-color: var(--gold);
          color: var(--pitch-night);
          font-weight: 600;
        }
        .sim-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        .chart-wrap {
          margin-top: 1.75rem;
          border-top: 1px solid rgba(242, 240, 228, 0.08);
          padding-top: 1.25rem;
        }
      `}</style>
    </div>
  )
}
