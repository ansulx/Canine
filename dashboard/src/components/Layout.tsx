import { Outlet, NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Brain, Users, Zap, BarChart3,
  Shield, Activity
} from 'lucide-react'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/training', icon: Brain, label: 'Training' },
  { to: '/clients', icon: Users, label: 'Clients' },
  { to: '/predict', icon: Zap, label: 'Predict' },
  { to: '/analytics', icon: BarChart3, label: 'Analytics' },
]

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-sidebar flex flex-col shrink-0">
        {/* Logo */}
        <div className="px-6 py-5 flex items-center gap-3 border-b border-white/10">
          <div className="w-8 h-8 rounded-lg bg-primary-500 flex items-center justify-center">
            <Shield className="w-4.5 h-4.5 text-white" />
          </div>
          <div>
            <h1 className="text-white font-semibold text-[15px] tracking-tight">FedStable</h1>
            <p className="text-slate-500 text-[10px] tracking-widest uppercase">Risk Platform</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-sidebar-active text-white'
                    : 'text-slate-400 hover:text-white hover:bg-sidebar-hover'
                }`
              }
            >
              <Icon className="w-[18px] h-[18px]" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* System Status */}
        <div className="px-4 py-4 border-t border-white/10">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Activity className="w-3.5 h-3.5 text-success animate-pulse-dot" />
            <span>System Operational</span>
          </div>
          <p className="text-[10px] text-slate-600 mt-1 ml-5.5">v0.1.0 · Simulation Mode</p>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto bg-surface">
        <div className="p-8 max-w-7xl mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
