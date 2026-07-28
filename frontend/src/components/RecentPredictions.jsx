export default function RecentPredictions({ items }) {
  return (
    <div className="card">
      <div className="card-title">Recent predictions</div>
      {items.length === 0 && <p className="empty mono">No predictions logged yet this session.</p>}
      <ul className="feed">
        {items.map((p, i) => (
          <li key={i} className="feed-row">
            <span className="feed-teams">{p.team_a} <span className="vs">v</span> {p.team_b}</span>
            <span className="feed-probs mono">
              <span style={{ color: 'var(--win-a)' }}>{Math.round(p.prob_win_a * 100)}</span>
              <span className="sep">/</span>
              <span style={{ color: 'var(--draw)' }}>{Math.round(p.prob_draw * 100)}</span>
              <span className="sep">/</span>
              <span style={{ color: 'var(--win-b)' }}>{Math.round(p.prob_win_b * 100)}</span>
            </span>
          </li>
        ))}
      </ul>
      <style>{`
        .empty { color: var(--chalk-dim); font-size: 0.85rem; }
        .feed { list-style: none; margin: 0; padding: 0; }
        .feed-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.6rem 0;
          border-top: 1px solid rgba(242, 240, 228, 0.08);
          font-size: 0.85rem;
        }
        .feed-row:first-child { border-top: none; }
        .feed-teams { color: var(--chalk); }
        .vs { color: var(--chalk-dim); font-size: 0.75rem; }
        .feed-probs { font-size: 0.78rem; }
        .sep { color: var(--chalk-dim); margin: 0 0.2rem; }
      `}</style>
    </div>
  )
}
