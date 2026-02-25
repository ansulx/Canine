import { useState } from 'react'
import { Zap, Clock, TrendingUp, AlertCircle, ChevronDown, History, Loader2 } from 'lucide-react'

interface PredictionResult {
  probabilities: number[]; horizon: string; model_version: string; timestamp: string
}

const featureLabels = [
  'Price', 'Peg Deviation (%)', 'Volume 24h', 'Volatility 24h',
  'Supply', 'Reserve Proxy', 'Flow Ratio', 'Momentum'
]

const presets: Record<string, number[]> = {
  'Normal Market': [1.0001, 0.01, 45000000, 0.003, 3200000000, 0.98, 0.52, 0.01],
  'Mild Stress': [0.9970, -0.30, 120000000, 0.012, 3100000000, 0.92, 0.65, -0.15],
  'High Volatility': [0.9920, -0.80, 350000000, 0.025, 2800000000, 0.85, 0.78, -0.35],
  'Crisis Scenario': [0.9750, -2.50, 800000000, 0.055, 2200000000, 0.72, 0.92, -0.60],
}

export default function Predict() {
  const [features, setFeatures] = useState<number[]>(presets['Normal Market'])
  const [horizon, setHorizon] = useState('6h')
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<{ prob: number; horizon: string; time: string; preset?: string }[]>([])
  const [activePreset, setActivePreset] = useState('Normal Market')

  const predict = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: [features], horizon }),
      })
      const data: PredictionResult = await res.json()
      setResult(data)
      setHistory(prev => [{ prob: data.probabilities[0], horizon, time: new Date().toLocaleTimeString(), preset: activePreset }, ...prev.slice(0, 9)])
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  const riskLevel = (p: number) => p > 0.7 ? 'Critical' : p > 0.5 ? 'High' : p > 0.3 ? 'Elevated' : p > 0.15 ? 'Moderate' : 'Low'
  const riskColor = (p: number) => p > 0.7 ? 'text-danger' : p > 0.5 ? 'text-orange-500' : p > 0.3 ? 'text-warning' : p > 0.15 ? 'text-amber-400' : 'text-success'
  const riskBg = (p: number) => p > 0.7 ? 'bg-danger/10 border-danger/20' : p > 0.5 ? 'bg-orange-50 border-orange-200' : p > 0.3 ? 'bg-warning/10 border-warning/20' : p > 0.15 ? 'bg-amber-50 border-amber-200' : 'bg-success/10 border-success/20'

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Risk Prediction</h1>
        <p className="text-text-secondary text-sm mt-1">Run stablecoin depeg probability predictions.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Panel */}
        <div className="lg:col-span-2 space-y-4">
          {/* Presets */}
          <div className="bg-surface-raised rounded-xl border border-border p-5 animate-fade-in">
            <h3 className="font-semibold text-sm mb-3">Scenario Presets</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {Object.keys(presets).map(name => (
                <button key={name} onClick={() => { setFeatures(presets[name]); setActivePreset(name) }}
                  className={`px-3 py-2 rounded-lg text-xs font-medium border transition-all ${
                    activePreset === name
                      ? 'bg-primary-50 border-primary-200 text-primary-700'
                      : 'bg-surface border-border text-text-secondary hover:border-primary-200'
                  }`}>
                  {name}
                </button>
              ))}
            </div>
          </div>

          {/* Feature Inputs */}
          <div className="bg-surface-raised rounded-xl border border-border p-5 animate-fade-in">
            <h3 className="font-semibold text-sm mb-4">Feature Vector</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {featureLabels.map((label, i) => (
                <div key={i}>
                  <label className="text-[10px] font-medium text-text-muted uppercase tracking-wider">{label}</label>
                  <input type="number" step="any" value={features[i]}
                    onChange={e => { const f = [...features]; f[i] = +e.target.value; setFeatures(f) }}
                    className="mt-1 w-full px-3 py-2 rounded-lg border border-border text-sm bg-surface focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 font-mono" />
                </div>
              ))}
            </div>
          </div>

          {/* Horizon + Run */}
          <div className="flex items-center gap-3">
            <div className="relative flex-1">
              <select value={horizon} onChange={e => setHorizon(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg border border-border text-sm bg-surface-raised focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 appearance-none">
                <option value="1h">1-Hour Horizon</option>
                <option value="6h">6-Hour Horizon</option>
                <option value="24h">24-Hour Horizon</option>
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none" />
            </div>
            <button onClick={predict} disabled={loading}
              className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-primary-600 text-white text-sm font-medium hover:bg-primary-700 transition-colors shadow-sm disabled:opacity-50">
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              Run Prediction
            </button>
          </div>
        </div>

        {/* Result Panel */}
        <div className="space-y-4">
          {result ? (
            <div className={`rounded-xl border-2 p-6 animate-fade-in ${riskBg(result.probabilities[0])}`}>
              <div className="text-center">
                <p className="text-xs font-medium text-text-secondary uppercase tracking-wider mb-2">Depeg Probability</p>
                <p className={`text-5xl font-bold tracking-tight ${riskColor(result.probabilities[0])}`}>
                  {(result.probabilities[0] * 100).toFixed(1)}%
                </p>
                <div className={`inline-flex items-center gap-1.5 mt-3 px-3 py-1 rounded-full text-xs font-semibold ${riskColor(result.probabilities[0])} bg-white/60`}>
                  <AlertCircle className="w-3.5 h-3.5" />
                  {riskLevel(result.probabilities[0])} Risk
                </div>
              </div>
              <div className="mt-5 space-y-2 text-xs text-text-secondary">
                <div className="flex justify-between">
                  <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> Horizon</span>
                  <span className="font-medium text-text-primary">{result.horizon}</span>
                </div>
                <div className="flex justify-between">
                  <span className="flex items-center gap-1"><TrendingUp className="w-3 h-3" /> Model</span>
                  <span className="font-medium text-text-primary">{result.model_version}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-border bg-surface-raised p-6 text-center animate-fade-in">
              <Zap className="w-8 h-8 text-text-muted mx-auto mb-2" />
              <p className="text-sm text-text-secondary">Run a prediction to see results</p>
            </div>
          )}

          {/* History */}
          {history.length > 0 && (
            <div className="bg-surface-raised rounded-xl border border-border p-5 animate-fade-in">
              <div className="flex items-center gap-2 mb-3">
                <History className="w-4 h-4 text-text-secondary" />
                <h3 className="font-semibold text-sm">Recent</h3>
              </div>
              <div className="space-y-2">
                {history.map((h, i) => (
                  <div key={i} className="flex items-center justify-between text-xs p-2 rounded-lg bg-surface">
                    <div>
                      <span className={`font-semibold ${riskColor(h.prob)}`}>{(h.prob * 100).toFixed(1)}%</span>
                      <span className="text-text-muted ml-2">{h.horizon}</span>
                    </div>
                    <span className="text-text-muted">{h.time}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
