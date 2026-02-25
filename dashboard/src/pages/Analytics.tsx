import { useEffect, useState } from 'react'
import { BarChart3, TrendingDown, Layers, Play, Loader2 } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

interface SimResult {
  episodes: { episode: number; total_reward: number; steps: number; final_deviation: number; trajectory: any[] }[]
  summary: { avg_reward: number; avg_steps: number; avg_final_deviation: number }
}

export default function Analytics() {
  const [metrics, setMetrics] = useState<any[]>([])
  const [simResult, setSimResult] = useState<SimResult | null>(null)
  const [simLoading, setSimLoading] = useState(false)
  const [simConfig, setSimConfig] = useState({ episodes: 10, max_steps: 50, action_strategy: 'random' })

  useEffect(() => {
    fetch('/api/dashboard/stats').then(r => r.json()).then(d => setMetrics(d.recent_metrics || []))
  }, [])

  const runSim = async () => {
    setSimLoading(true)
    const res = await fetch('/api/simulation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(simConfig),
    })
    setSimResult(await res.json())
    setSimLoading(false)
  }

  const radarData = [
    { metric: 'Accuracy', value: 84.7, fullMark: 100 },
    { metric: 'Precision', value: 79.2, fullMark: 100 },
    { metric: 'Recall', value: 88.1, fullMark: 100 },
    { metric: 'F1', value: 83.4, fullMark: 100 },
    { metric: 'Calibration', value: 76.8, fullMark: 100 },
    { metric: 'Robustness', value: 71.5, fullMark: 100 },
  ]

  const comparisonData = [
    { method: 'Local Only', auc: 0.72, brier: 0.28, comm: 0 },
    { method: 'FedAvg', auc: 0.85, brier: 0.15, comm: 120 },
    { method: 'FedProx', auc: 0.86, brier: 0.14, comm: 130 },
    { method: 'Centralized', auc: 0.91, brier: 0.09, comm: 0 },
  ]

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Analytics</h1>
        <p className="text-text-secondary text-sm mt-1">Model performance analysis, comparisons, and simulations.</p>
      </div>

      {/* Performance Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Loss over time */}
        <div className="bg-surface-raised rounded-xl border border-border p-6 animate-fade-in">
          <div className="flex items-center gap-2 mb-4">
            <TrendingDown className="w-4 h-4 text-text-secondary" />
            <h2 className="font-semibold text-sm">Training Loss</h2>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} tickFormatter={v => v.slice(5)} />
              <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }} />
              <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Radar */}
        <div className="bg-surface-raised rounded-xl border border-border p-6 animate-fade-in">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-4 h-4 text-text-secondary" />
            <h2 className="font-semibold text-sm">Model Quality</h2>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
              <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.15} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Method Comparison */}
      <div className="bg-surface-raised rounded-xl border border-border p-6 animate-fade-in">
        <div className="flex items-center gap-2 mb-4">
          <Layers className="w-4 h-4 text-text-secondary" />
          <h2 className="font-semibold text-sm">Method Comparison</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-text-muted uppercase tracking-wider border-b border-border">
                <th className="pb-3 pr-4">Method</th>
                <th className="pb-3 pr-4">AUC-ROC</th>
                <th className="pb-3 pr-4">Brier Score</th>
                <th className="pb-3 pr-4">Comm. Cost (MB)</th>
                <th className="pb-3">Performance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {comparisonData.map(row => (
                <tr key={row.method} className="hover:bg-surface transition-colors">
                  <td className="py-3 pr-4 font-medium">{row.method}</td>
                  <td className="py-3 pr-4 font-mono">{row.auc.toFixed(2)}</td>
                  <td className="py-3 pr-4 font-mono">{row.brier.toFixed(2)}</td>
                  <td className="py-3 pr-4 font-mono">{row.comm || '—'}</td>
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 bg-border rounded-full overflow-hidden max-w-24">
                        <div className="h-full bg-primary-500 rounded-full" style={{ width: `${row.auc * 100}%` }} />
                      </div>
                      <span className="text-xs text-text-muted">{(row.auc * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* RL Simulation */}
      <div className="bg-surface-raised rounded-xl border border-border p-6 animate-fade-in">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="font-semibold text-sm">Intervention Simulation (Pillar 3)</h2>
            <p className="text-xs text-text-secondary mt-0.5">Run the stablecoin environment with different strategies.</p>
          </div>
          <div className="flex items-center gap-3">
            <select value={simConfig.action_strategy} onChange={e => setSimConfig({ ...simConfig, action_strategy: e.target.value })}
              className="px-3 py-1.5 rounded-lg border border-border text-xs bg-surface focus:outline-none focus:ring-2 focus:ring-primary-500/20">
              <option value="random">Random Policy</option>
              <option value="greedy">Greedy Policy</option>
            </select>
            <input type="number" value={simConfig.episodes} min={1} max={100}
              onChange={e => setSimConfig({ ...simConfig, episodes: +e.target.value })}
              className="w-20 px-3 py-1.5 rounded-lg border border-border text-xs bg-surface focus:outline-none" />
            <button onClick={runSim} disabled={simLoading}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-primary-600 text-white text-xs font-medium hover:bg-primary-700 transition-colors disabled:opacity-50">
              {simLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
              Run
            </button>
          </div>
        </div>

        {simResult ? (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-4 rounded-lg bg-surface">
                <p className="text-lg font-semibold">{simResult.summary.avg_reward.toFixed(2)}</p>
                <p className="text-[10px] text-text-muted uppercase tracking-wider mt-1">Avg Reward</p>
              </div>
              <div className="text-center p-4 rounded-lg bg-surface">
                <p className="text-lg font-semibold">{simResult.summary.avg_steps.toFixed(0)}</p>
                <p className="text-[10px] text-text-muted uppercase tracking-wider mt-1">Avg Steps</p>
              </div>
              <div className="text-center p-4 rounded-lg bg-surface">
                <p className="text-lg font-semibold">{simResult.summary.avg_final_deviation.toFixed(3)}</p>
                <p className="text-[10px] text-text-muted uppercase tracking-wider mt-1">Avg Final Dev</p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={simResult.episodes}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="episode" tick={{ fontSize: 10 }} tickLine={false} />
                <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }} />
                <Bar dataKey="total_reward" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-40 flex items-center justify-center text-text-muted text-sm">
            Run a simulation to see results
          </div>
        )}
      </div>
    </div>
  )
}
