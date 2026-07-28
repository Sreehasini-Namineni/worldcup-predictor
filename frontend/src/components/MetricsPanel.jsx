export default function MetricsPanel({ prediction }) {
  if (!prediction) {
    return (
      <div className="card">
        <div className="card-title">Expected value</div>
        <p className="empty mono">Run a prediction to see EV and confidence metrics.</p>
        <style>{`.empty { color: var(--chalk-dim); font-size: 0.85rem; }`}</style>
      </div>
    )
  }

  const rows = [
    { label: `${prediction.team_a} win`, value: prediction.expected_value.win_a },
    { label: 'Draw', value: prediction.expected_value.draw },
    { label: `${prediction.team_b} win`, value: prediction.expected_value.win_b },
  ]

  return (
    <div className="card">
      <div className="card-title">Expected value vs. flat market</div>
      <p className="subtext">EV per $1 staked against a naive 33/33/33 baseline — positive means the model sees an edge.</p>
      <table className="ev-table">
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td className="mono">{r.label}</td>
              <td className={`mono ev-value ${r.value >= 0 ? 'pos' : 'neg'}`}>
                {r.value >= 0 ? '+' : ''}{(r.value * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <style>{`
        .subtext {
          font-size: 0.8rem;
          color: var(--chalk-dim);
          margin: -0.5rem 0 1rem 0;
          line-height: 1.4;
        }
        .ev-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.85rem;
        }
        .ev-table td {
          padding: 0.55rem 0;
          border-top: 1px solid rgba(242, 240, 228, 0.08);
        }
        .ev-value { text-align: right; font-weight: 600; }
        .ev-value.pos { color: var(--win-a); }
        .ev-value.neg { color: var(--danger); }
      `}</style>
    </div>
  )
}
