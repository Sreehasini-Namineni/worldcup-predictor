export default function Scoreboard({ prediction, teamA, teamB }) {
  const hasData = !!prediction

  const pctA = hasData ? Math.round(prediction.prob_win_a * 100) : 33
  const pctDraw = hasData ? Math.round(prediction.prob_draw * 100) : 34
  const pctB = hasData ? Math.round(prediction.prob_win_b * 100) : 33

  return (
    <div className="scoreboard">
      <div className="scoreboard-row">
        <div className="scoreboard-team">
          <div className="scoreboard-label">{teamA || 'TEAM A'}</div>
          <div className="scoreboard-digits" style={{ color: 'var(--win-a)' }}>
            {String(pctA).padStart(2, '0')}
            <span className="scoreboard-pct">%</span>
          </div>
        </div>

        <div className="scoreboard-mid">
          <div className="scoreboard-vs mono">VS</div>
          <div className="scoreboard-draw">
            <span className="scoreboard-draw-num">{pctDraw}%</span>
            <span className="eyebrow">draw</span>
          </div>
        </div>

        <div className="scoreboard-team">
          <div className="scoreboard-label">{teamB || 'TEAM B'}</div>
          <div className="scoreboard-digits" style={{ color: 'var(--win-b)' }}>
            {String(pctB).padStart(2, '0')}
            <span className="scoreboard-pct">%</span>
          </div>
        </div>
      </div>

      <div className="scoreboard-bar" aria-hidden="true">
        <div className="scoreboard-bar-seg" style={{ width: `${pctA}%`, background: 'var(--win-a)' }} />
        <div className="scoreboard-bar-seg" style={{ width: `${pctDraw}%`, background: 'var(--draw)' }} />
        <div className="scoreboard-bar-seg" style={{ width: `${pctB}%`, background: 'var(--win-b)' }} />
      </div>

      {hasData && (
        <div className="scoreboard-footer mono">
          confidence {(prediction.confidence * 100).toFixed(0)}%
          &nbsp;·&nbsp; model v{prediction.model_version}
        </div>
      )}

      <style>{`
        .scoreboard {
          background: linear-gradient(180deg, var(--pitch-mid) 0%, var(--pitch-night) 100%);
          border: var(--border);
          border-radius: var(--radius);
          padding: 2rem clamp(1rem, 4vw, 3rem);
        }
        .scoreboard-row {
          display: grid;
          grid-template-columns: 1fr auto 1fr;
          align-items: center;
          gap: 1rem;
        }
        .scoreboard-team {
          text-align: center;
        }
        .scoreboard-label {
          font-family: var(--font-mono);
          font-size: clamp(0.7rem, 1.6vw, 0.95rem);
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--chalk-dim);
          margin-bottom: 0.4rem;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .scoreboard-digits {
          font-family: var(--font-display);
          font-size: clamp(2.6rem, 9vw, 5.5rem);
          line-height: 1;
        }
        .scoreboard-pct {
          font-size: 0.4em;
          margin-left: 0.15em;
          vertical-align: super;
        }
        .scoreboard-mid {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.5rem;
          padding: 0 clamp(0.5rem, 2vw, 2rem);
        }
        .scoreboard-vs {
          color: var(--gold);
          font-size: 0.85rem;
          letter-spacing: 0.2em;
        }
        .scoreboard-draw {
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        .scoreboard-draw-num {
          font-family: var(--font-mono);
          font-size: 1.1rem;
          color: var(--draw);
        }
        .scoreboard-bar {
          display: flex;
          height: 6px;
          border-radius: 3px;
          overflow: hidden;
          margin-top: 1.75rem;
        }
        .scoreboard-bar-seg {
          transition: width 0.4s ease;
        }
        .scoreboard-footer {
          margin-top: 0.85rem;
          text-align: center;
          font-size: 0.75rem;
          color: var(--chalk-dim);
        }
      `}</style>
    </div>
  )
}
