import { useEffect, useState } from 'react'
import { Plus, MoreVertical, Building2, Globe, Database, UserCheck, UserX, Trash2 } from 'lucide-react'

interface Client {
  id: string; name: string; type: string; status: string; samples: number;
  stablecoin: string; region: string; joined: string; last_seen: string; contribution_score: number
}

const typeIcons: Record<string, React.ElementType> = {
  exchange: Building2,
  defi_protocol: Globe,
  custodian: Database,
}
const typeLabels: Record<string, string> = {
  exchange: 'Exchange',
  defi_protocol: 'DeFi Protocol',
  custodian: 'Custodian',
}

export default function Clients() {
  const [clients, setClients] = useState<Client[]>([])
  const [showAdd, setShowAdd] = useState(false)
  const [form, setForm] = useState({ name: '', type: 'exchange', stablecoin: 'USDC', region: 'Global' })
  const [menuOpen, setMenuOpen] = useState<string | null>(null)

  const load = () => fetch('/api/clients').then(r => r.json()).then(d => setClients(d.clients))
  useEffect(() => { load() }, [])

  const addClient = async () => {
    if (!form.name.trim()) return
    await fetch('/api/clients', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) })
    setForm({ name: '', type: 'exchange', stablecoin: 'USDC', region: 'Global' })
    setShowAdd(false)
    load()
  }

  const deleteClient = async (id: string) => {
    await fetch(`/api/clients/${id}`, { method: 'DELETE' })
    setMenuOpen(null)
    load()
  }

  const toggleStatus = async (c: Client) => {
    const newStatus = c.status === 'active' ? 'idle' : 'active'
    await fetch(`/api/clients/${c.id}/status?status=${newStatus}`, { method: 'PATCH' })
    setMenuOpen(null)
    load()
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Federation Clients</h1>
          <p className="text-text-secondary text-sm mt-1">Manage participating institutions and data silos.</p>
        </div>
        <button onClick={() => setShowAdd(!showAdd)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary-600 text-white text-sm font-medium hover:bg-primary-700 transition-colors shadow-sm">
          <Plus className="w-4 h-4" /> Add Client
        </button>
      </div>

      {/* Add Client Form */}
      {showAdd && (
        <div className="bg-surface-raised rounded-xl border border-border p-6 animate-fade-in">
          <h3 className="font-semibold text-sm mb-4">Register New Client</h3>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <input placeholder="Institution name" value={form.name}
              onChange={e => setForm({ ...form, name: e.target.value })}
              className="px-3 py-2 rounded-lg border border-border text-sm bg-surface focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500" />
            <select value={form.type} onChange={e => setForm({ ...form, type: e.target.value })}
              className="px-3 py-2 rounded-lg border border-border text-sm bg-surface focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500">
              <option value="exchange">Exchange</option>
              <option value="defi_protocol">DeFi Protocol</option>
              <option value="custodian">Custodian</option>
            </select>
            <select value={form.stablecoin} onChange={e => setForm({ ...form, stablecoin: e.target.value })}
              className="px-3 py-2 rounded-lg border border-border text-sm bg-surface focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500">
              <option value="USDC">USDC</option>
              <option value="USDT">USDT</option>
              <option value="DAI">DAI</option>
            </select>
            <button onClick={addClient}
              className="px-4 py-2 rounded-lg bg-primary-600 text-white text-sm font-medium hover:bg-primary-700 transition-colors">
              Register
            </button>
          </div>
        </div>
      )}

      {/* Client Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {clients.map(c => {
          const Icon = typeIcons[c.type] || Building2
          return (
            <div key={c.id} className="bg-surface-raised rounded-xl border border-border p-5 animate-fade-in hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    c.status === 'active' ? 'bg-primary-50 text-primary-600' : 'bg-slate-100 text-text-muted'
                  }`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-sm">{c.name}</h3>
                    <p className="text-xs text-text-secondary">{typeLabels[c.type] || c.type} · {c.region}</p>
                  </div>
                </div>
                <div className="relative">
                  <button onClick={() => setMenuOpen(menuOpen === c.id ? null : c.id)}
                    className="p-1.5 rounded-md hover:bg-surface transition-colors">
                    <MoreVertical className="w-4 h-4 text-text-muted" />
                  </button>
                  {menuOpen === c.id && (
                    <div className="absolute right-0 top-8 w-44 bg-surface-raised rounded-lg border border-border shadow-lg z-10 py-1">
                      <button onClick={() => toggleStatus(c)}
                        className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-surface transition-colors">
                        {c.status === 'active' ? <UserX className="w-3.5 h-3.5" /> : <UserCheck className="w-3.5 h-3.5" />}
                        {c.status === 'active' ? 'Set Idle' : 'Activate'}
                      </button>
                      <button onClick={() => deleteClient(c.id)}
                        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-danger hover:bg-surface transition-colors">
                        <Trash2 className="w-3.5 h-3.5" /> Remove
                      </button>
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-4 grid grid-cols-3 gap-3">
                <div className="text-center p-2 rounded-lg bg-surface">
                  <p className="text-sm font-semibold">{c.samples.toLocaleString()}</p>
                  <p className="text-[10px] text-text-muted">Samples</p>
                </div>
                <div className="text-center p-2 rounded-lg bg-surface">
                  <p className="text-sm font-semibold">{c.stablecoin}</p>
                  <p className="text-[10px] text-text-muted">Asset</p>
                </div>
                <div className="text-center p-2 rounded-lg bg-surface">
                  <p className="text-sm font-semibold">{(c.contribution_score * 100).toFixed(0)}%</p>
                  <p className="text-[10px] text-text-muted">Score</p>
                </div>
              </div>

              <div className="mt-3 flex items-center justify-between">
                <span className={`flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-full ${
                  c.status === 'active' ? 'bg-success/10 text-success' : 'bg-slate-100 text-text-muted'
                }`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${c.status === 'active' ? 'bg-success' : 'bg-text-muted'}`} />
                  {c.status}
                </span>
                <span className="text-[10px] text-text-muted">
                  Joined {new Date(c.joined).toLocaleDateString()}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
