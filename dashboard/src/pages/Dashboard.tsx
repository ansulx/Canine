import { useEffect, useState } from 'react'
import {
  Users, Database, Brain, Activity,
  ArrowUpRight, ArrowDownRight, AlertTriangle, CheckCircle2, Clock
} from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

interface DashboardStats {
  total_clients: number
  active_clients: number
  total_samples: number
  model_parameters: number
  training_rounds_completed: number
  predictions_served: number
  avg_contribution: number
  recent_metrics: { date: string; auc_roc: number; loss: number; predictions: number }[]
  risk_alerts: { id: string; severity: string; message: string; timestamp: string }[]
  pillars: { id: number; name: string; status: string; accuracy: number | null; description: string }[]
}

function StatCard({ icon: Icon, label, value, sub, trend, color }: {
  icon: React.ElementType; label: string; value: string | number; sub?: string;
  trend?: { value: string; up: boolean }; color: string
}) {
  return (
    <div className="bg-surface-raised rounded-xl border border-border p-5 animate-fade-in">
      <div className="flex items-start justify-between">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${color}`}>
          <Icon className="w-5 h-5" />
        </div>
        {trend && (
          <span className={`flex items-center text-xs font-medium ${trend.up ? 'text-success' : 'text-danger'}`}>
            {trend.up ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
            {trend.value}
          </span>
        )}
      </div>
      <p className="mt-3 text-2xl font-semibold tracking-tight">{value}</p>
      <p className="text-sm text-text-secondary mt-0.5">{label}</p>
      {sub && <p className="text-xs text-text-muted mt-1">{sub}</p>}
    </div>
  )
}

function PillarCard({ pillar }: { pillar: DashboardStats['pillars'][0] }) {
  const statusColors: Record<string, string> = {
    active: 'bg-success/10 text-success',
    development: 'bg-warning/10 text-warning',
    planned: 'bg-text-muted/10 text-text-muted',
  }
  return (
    <div className="bg-surface-raised rounded-xl border border-border p-5 animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono text-text-muted">PILLAR {pillar.id}</span>
        <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${statusColors[pillar.status] || statusColors.planned}`}>
          {pillar.status}
        </span>
      </div>
      <h3 className="font-semibold text-sm">{pillar.name}</h3>
      <p className="text-xs text-text-secondary mt-1 leading-relaxed">{pillar.description}</p>
      {pillar.accuracy !== null && (
        <div className="mt-3 flex items-center gap-2">
          <div className="flex-1 h-1.5 bg-border rounded-full overflow-hidden">
            <div className="h-full bg-primary-500 rounded-full" style={{ width: `${pillar.accuracy * 100}%` }} />
          </div>
          <span className="text-xs font-medium text-primary-600">{(pillar.accuracy * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null)

  useEffect(() => {
    fetch('/api/dashboard/stats').then(r => r.json()).then(setStats)
  }, [])

  if (!stats) return <div className="flex items-center justify-center h-64"><div className="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" /></div>

  const fmt = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n)

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-text-secondary text-sm mt-1">Federated learning platform overview and system health.</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Users} label="Active Clients" value={stats.active_clients}
          sub={`${stats.total_clients} total registered`}
          trend={{ value: '+1 this week', up: true }}
          color="bg-primary-50 text-primary-600" />
        <StatCard icon={Database} label="Total Samples" value={fmt(stats.total_samples)}
          sub="Across all silos"
          trend={{ value: '+12%', up: true }}
          color="bg-emerald-50 text-emerald-600" />
        <StatCard icon={Brain} label="Model Parameters" value={fmt(stats.model_parameters)}
          sub="DepegPredictor LSTM"
          color="bg-violet-50 text-violet-600" />
        <StatCard icon={Activity} label="Predictions Served" value={stats.predictions_served}
          sub="Real-time inference"
          color="bg-amber-50 text-amber-600" />
      </div>

      {/* Chart + Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* AUC Chart */}
        <div className="lg:col-span-2 bg-surface-raised rounded-xl border border-border p-6 animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="font-semibold text-sm">Model Performance</h2>
              <p className="text-xs text-text-secondary mt-0.5">AUC-ROC over training period</p>
            </div>
            <span className="text-xs font-mono text-text-muted">Last 30 days</span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={stats.recent_metrics}>
              <defs>
                <linearGradient id="aucGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickLine={false} axisLine={false}
                tickFormatter={v => v.slice(5)} />
              <YAxis domain={[0.7, 1]} tick={{ fontSize: 10 }} tickLine={false} axisLine={false}
                tickFormatter={v => v.toFixed(2)} />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
                formatter={(v: any) => [Number(v).toFixed(4), 'AUC-ROC']} />
              <Area type="monotone" dataKey="auc_roc" stroke="#6366f1" fill="url(#aucGrad)" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Alerts */}
        <div className="bg-surface-raised rounded-xl border border-border p-6 animate-fade-in">
          <h2 className="font-semibold text-sm mb-4">Risk Alerts</h2>
          <div className="space-y-3">
            {stats.risk_alerts.map(a => (
              <div key={a.id} className="flex items-start gap-3 p-3 rounded-lg bg-surface">
                {a.severity === 'medium' ? (
                  <AlertTriangle className="w-4 h-4 text-warning shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle2 className="w-4 h-4 text-text-muted shrink-0 mt-0.5" />
                )}
                <div>
                  <p className="text-xs leading-relaxed">{a.message}</p>
                  <p className="text-[10px] text-text-muted mt-1 flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {new Date(a.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Pillars */}
      <div>
        <h2 className="font-semibold text-sm mb-4">System Pillars</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {stats.pillars.map(p => <PillarCard key={p.id} pillar={p} />)}
        </div>
      </div>
    </div>
  )
}
