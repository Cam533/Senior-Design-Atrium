import { Link } from 'react-router-dom'
import atriumIcon from '../pics/atrium_icon.png'

const features = [
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
        <circle cx="12" cy="10" r="3" />
      </svg>
    ),
    title: 'Identify Vacant Lots',
    text: 'Surface vacant and underused lots using city-level building and land data.',
  },
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
    title: 'AI-Powered Insights',
    text: 'Get zoning- and code-aware guidance through a conversational AI interface.',
  },
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
      </svg>
    ),
    title: 'Smart Recommendations',
    text: 'Recommendations that balance demand, feasibility, sustainability, and equity.',
  },
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    ),
    title: 'Connect Stakeholders',
    text: 'Link realty groups, architects, and contractors based on relevant experience.',
  },
]

const builtItems = [
  'Interactive map with lot boundaries and vacancy indicators for Philadelphia',
  'AI-powered lot-level chat grounded in zoning and building codes',
  'Census and demographic data at the lot level',
  'Environmental and location scores including walkability, transit, and recreation',
  'User accounts with saved lots, projects, and notification preferences',
]

export default function About() {
  return (
    <div style={{ fontFamily: 'Inter, system-ui, -apple-system, sans-serif', color: '#0f172a' }}>

      {/* Hero */}
      <div style={{
        textAlign: 'center',
        padding: '64px 24px 56px',
        background: 'linear-gradient(180deg, #f8fafc 0%, #fff 100%)',
      }}>
        <img src={atriumIcon} alt="Atrium" style={{ width: 64, height: 64, borderRadius: 14, marginBottom: 20 }} />
        <h1 style={{
          fontSize: 42,
          fontWeight: 800,
          letterSpacing: '-0.03em',
          margin: '0 0 12px',
          fontFamily: "'Montserrat', Inter, system-ui, sans-serif",
          textTransform: 'uppercase',
        }}>
          Atrium
        </h1>
        <p style={{
          fontSize: 18,
          color: '#475569',
          lineHeight: 1.6,
          maxWidth: 540,
          margin: '0 auto 32px',
        }}>
          An AI-powered platform that helps cities, planners, developers, and residents
          understand and activate vacant urban lots.
        </p>
        <Link to="/" style={{
          display: 'inline-block',
          padding: '14px 32px',
          background: '#0f172a',
          color: '#fff',
          borderRadius: 10,
          fontWeight: 600,
          fontSize: 15,
          textDecoration: 'none',
          transition: 'background 140ms ease, transform 100ms ease',
        }}>
          Explore the Map
        </Link>
      </div>

      {/* Problem banner */}
      <div style={{
        background: '#0f172a',
        color: '#fff',
        padding: '40px 24px',
        textAlign: 'center',
      }}>
        <div style={{ maxWidth: 620, margin: '0 auto' }}>
          <h2 style={{
            fontSize: 14,
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: '#94a3b8',
            marginBottom: 12,
          }}>
            The Problem
          </h2>
          <p style={{ fontSize: 17, lineHeight: 1.7, color: '#e2e8f0', margin: 0 }}>
            Cities contain many vacant or underutilized plots that could support housing, green space,
            or community infrastructure. Understanding development feasibility requires navigating
            fragmented data sources, complex zoning codes, and limited planning expertise.
          </p>
        </div>
      </div>

      {/* Feature cards */}
      <div style={{ maxWidth: 880, margin: '0 auto', padding: '56px 24px 48px' }}>
        <h2 style={{
          textAlign: 'center',
          fontSize: 24,
          fontWeight: 800,
          marginBottom: 36,
          letterSpacing: '-0.01em',
        }}>
          What Atrium Does
        </h2>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 20,
        }}>
          {features.map((f, i) => (
            <div key={i} style={{
              background: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: 12,
              padding: '24px 20px',
              textAlign: 'center',
            }}>
              <div style={{ marginBottom: 12 }}>{f.icon}</div>
              <h3 style={{ fontSize: 15, fontWeight: 700, margin: '0 0 8px', color: '#0f172a' }}>
                {f.title}
              </h3>
              <p style={{ fontSize: 13, lineHeight: 1.55, color: '#64748b', margin: 0 }}>
                {f.text}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* What we've built */}
      <div style={{
        background: '#f8fafc',
        borderTop: '1px solid #e2e8f0',
        borderBottom: '1px solid #e2e8f0',
        padding: '48px 24px',
      }}>
        <div style={{ maxWidth: 620, margin: '0 auto' }}>
          <h2 style={{
            textAlign: 'center',
            fontSize: 24,
            fontWeight: 800,
            marginBottom: 28,
            letterSpacing: '-0.01em',
          }}>
            What We've Built
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {builtItems.map((item, i) => (
              <div key={i} style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 12,
                background: '#fff',
                border: '1px solid #e6edf3',
                borderRadius: 10,
                padding: '14px 18px',
              }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginTop: 1 }}>
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                <span style={{ fontSize: 14, lineHeight: 1.5, color: '#334155' }}>{item}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer spacer */}
      <div style={{ height: 48 }} />
    </div>
  )
}
