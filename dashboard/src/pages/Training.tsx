import { useEffect, useRef, useState } from 'react'
import { Play, Square, Settings2, Loader2, Shield } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

interface RoundData {
  round: number; loss: number; auc_roc: number; brier_score: number;
  clients_participating: number; client_losses: Record<string, number>; dp_noise: number
}

export default function Training() {
  const [status, setStatus] = useState<string>('idle')
  const [rounds, setRounds] = useState<RoundData[]>([])
  const [config, setConfig] = useState({ pillar: 1, rounds: 20, clients_per_round: 3, local_epochs: 2, learning_rate: 0.001, dp_enabled: false })
  const [currentRound, setCurrentRound] = useState(0)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    fetch('/api/training/status').then(r => r.json()).then(data => {
      setStatus(data.status)
      setRounds(data.history || [])
      setCurrentRound(data.current_round || 0)
    })
  }, [])

  const connectWs = () => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/training`)
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'training_update') {
        setRounds(prev => [...prev, msg.data])
        setCurrentRound(msg.data.round)
      }
      if (msg.type === 'training_complete') {
        setStatus(msg.data.status)
      }
    }
    wsRef.current = ws
  }

  const startTraining = async () => {
    setRounds([])
    setCurrentRound(0)
    connectWs()
    const res = await fetch('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    if (res.ok) setStatus('training')
  }

  const stopTraining = async () => {
    await fetch('/api/training/stop', { method: 'POST' })
    setStatus('stopped')
    wsRef.current?.close()
  }

  const latestMetrics = rounds.length > 0 ? rounds[rounds.length - 1] : null
  const progress = config.rounds > 0 ? (currentRound / config.rounds) * 100 : 0

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Federated Training</h1>
        <p className="text-text-secondary text-sm mt-1">Configure and monitor federated learning rounds.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config Panel */}
        <div className="bg-surface-raised rounded-xl border border-border p-6 animate-fade-in">
          <div className="flex items-center gap-2 mb-5">
            <Settings2 className="w-4 h-4 text-text-secondary" />
            <h2 className="font-semibold text-sm">Configuration</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-xs font-medium text-text-secondary">Pillar</label>
              <select value={config.pillar} onChange={e => setConfig({ ...config, pillar: +e.target.value })}
                className="mt-1 w-full px-3 py-2 rounded-lg border border-border text-sm bg-surface focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500">
                <option value={1}>P1 — Depeg Prediction</option>
                <option value={2}>P2 — Contagion Analysis</option>
                <option value={3}>P3 — Intervention Policy</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-text-secondary">Rounds</label>
                <input type="number" value={config.rounds} min={1} max={500}
                  onChange={e => setConfig({ ...config, rounds: +e.target.value })}
                  className="mt-1 w-full px-3 py-2 rounded-lg border border-border text-sm bg-surface focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500" />
              </div>
              <div>
                <label className="text-xs font-medium text-text-secondary">Clients/Round</label>
                <input type="number" value={config.clients_per_round} min={1} max={20}
                  onChange={e => setConfig({ ...config, clients_per_round: +e.target.value })}
                  className="mt-1 w-full px-3 py-2 rounded-lg border border-border text-sm bg-surface focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500" />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-text-secondary">Local Epochs</label>
                <input type="number" value={config.local_epochs} min={1} max={10}
                  onChange={e => setConfig({ ...config, local_epochs: +e.target.value })}
                  className="mt-1 w-full px-3 py-2 rounded-lg border border-border text-sm bg-surface focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500" />
              </div>
              <div>
                <label className="text-xs font-medium text-text-secondary">Learning Rate</label>
                <input type="number" value={config.learning_rate} step={0.0001} min={0.0001}
                  onChange={e => setConfig({ ...config, learning_rate: +e.target.value })}
                  className="mt-1 w-full px-3 py-2 rounded-lg border border-border text-sm bg-surface focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500" />
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 rounded-lg bg-surface">
              <Shield className="w-4 h-4 text-primary-500" />
              <div className="flex-1">
                <p className="text-xs font-medium">Differential Privacy</p>
                <p className="text-[10px] text-text-muted">Add DP noise to gradients</p>
              </div>
              <button onClick={() => setConfig({ ...config, dp_enabled: !config.dp_enabled })}
                className={`w-9 h-5 rounded-full transition-colors ${config.dp_enabled ? 'bg-primary-500' : 'bg-border'}`}>
                <div className={`w-4 h-4 rounded-full bg-white shadow transition-transform ${config.dp_enabled ? 'translate-x-4.5' : 'translate-x-0.5'}`} />
              </button>
            </div>

            <button onClick={status === 'training' ? stopTraining : startTraining}
              disabled={status === 'training' && currentRound < 1}
              className={`w-full py-2.5 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all ${
                status === 'training'
                  ? 'bg-danger/10 text-danger hover:bg-danger/20'
                  : 'bg-primary-600 text-white hover:bg-primary-700 shadow-sm'
              }`}>
              {status === 'training' ? <><Square className="w-4 h-4" /> Stop Training</> : <><Play className="w-4 h-4" /> Start Training</>}
            </button>
          </div>
        </div>

        {/* Chart + Progress */}
        <div className="lg:col-span-2 space-y-4">
          {/* Progress Bar */}
          {status === 'training' && (
            <div className="bg-surface-raised rounded-xl border border-border p-5 animate-fade-in">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 text-primary-500 animate-spin" />
                  <span className="text-sm font-medium">Training in progress</span>
                </div>
                <span className="text-xs font-mono text-text-secondary">
                  Round {currentRound} / {config.rounds}
                </span>
              </div>
              <div className="h-2 bg-border rounded-full overflow-hidden">
                <div className="h-full bg-primary-500 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}

          {/* Charts */}
          <div className="bg-surface-raised rounded-xl border border-border p-6 animate-fade-in">
            <h2 className="font-semibold text-sm mb-4">Training Metrics</h2>
            {rounds.length === 0 ? (
              <div className="h-56 flex items-center justify-center text-text-muted text-sm">
                Start training to see live metrics
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={rounds}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="round" tick={{ fontSize: 10 }} tickLine={false} />
                  <YAxis yAxisId="left" domain={[0, 1]} tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} />
                  <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} name="Loss" />
                  <Line yAxisId="right" type="monotone" dataKey="auc_roc" stroke="#6366f1" strokeWidth={2} dot={false} name="AUC-ROC" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Latest Metrics */}
          {latestMetrics && (
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-surface-raised rounded-xl border border-border p-4 text-center animate-fade-in">
                <p className="text-xl font-semibold text-primary-600">{latestMetrics.auc_roc.toFixed(3)}</p>
                <p className="text-xs text-text-secondary mt-1">AUC-ROC</p>
              </div>
              <div className="bg-surface-raised rounded-xl border border-border p-4 text-center animate-fade-in">
                <p className="text-xl font-semibold text-danger">{latestMetrics.loss.toFixed(3)}</p>
                <p className="text-xs text-text-secondary mt-1">Loss</p>
              </div>
              <div className="bg-surface-raised rounded-xl border border-border p-4 text-center animate-fade-in">
                <p className="text-xl font-semibold text-amber-600">{latestMetrics.brier_score.toFixed(3)}</p>
                <p className="text-xs text-text-secondary mt-1">Brier Score</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
