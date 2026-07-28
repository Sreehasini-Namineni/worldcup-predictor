const BASE = import.meta.env.VITE_API_BASE_URL || '/api'

async function handle(res) {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

export const api = {
  health: () => fetch(`${BASE}/health`).then(handle),

  teams: () => fetch(`${BASE}/teams`).then(handle),

  predict: (payload) =>
    fetch(`${BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(handle),

  simulate: (payload) =>
    fetch(`${BASE}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(handle),

  retrain: () => fetch(`${BASE}/train`, { method: 'POST' }).then(handle),

  recentPredictions: (limit = 20) =>
    fetch(`${BASE}/predictions/recent?limit=${limit}`).then(handle),
}
