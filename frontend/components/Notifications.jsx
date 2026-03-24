import { useState, useEffect, useCallback, useRef } from 'react'
import { useAuth } from '../src/context/AuthContext'
import { Link } from 'react-router-dom'

const API_BASE = 'http://localhost:8000'

export function useUnreadCount() {
  const { user } = useAuth()
  const [count, setCount] = useState(0)

  const refresh = useCallback(async () => {
    if (!user) { setCount(0); return }
    try {
      const res = await fetch(`${API_BASE}/notifications/${user.id}/unread-count`)
      if (res.ok) {
        const data = await res.json()
        setCount(data.unread ?? 0)
      }
    } catch { /* backend may be offline */ }
  }, [user])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 30_000)
    return () => clearInterval(id)
  }, [refresh])

  return { count, refresh }
}

function timeAgo(dateStr) {
  const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

const popupStyles = {
  wrapper: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
  },
  dropdown: {
    position: 'absolute',
    top: 'calc(100% + 8px)',
    right: 0,
    width: 340,
    maxHeight: 420,
    background: '#fff',
    borderRadius: 12,
    boxShadow: '0 12px 40px rgba(15,23,42,0.18), 0 2px 8px rgba(15,23,42,0.08)',
    border: '1px solid #e2e8f0',
    zIndex: 9999,
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '14px 16px 10px',
    borderBottom: '1px solid #f1f5f9',
  },
  title: {
    fontSize: 15,
    fontWeight: 700,
    color: '#0f172a',
    margin: 0,
  },
  markAll: {
    background: 'none',
    border: 'none',
    color: '#2563eb',
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    padding: '2px 6px',
    borderRadius: 4,
  },
  list: {
    flex: 1,
    overflowY: 'auto',
    padding: '4px 0',
  },
  item: (isRead) => ({
    display: 'flex',
    alignItems: 'flex-start',
    gap: 10,
    padding: '10px 16px',
    cursor: isRead ? 'default' : 'pointer',
    opacity: isRead ? 0.55 : 1,
    borderLeft: isRead ? '3px solid transparent' : '3px solid #2563eb',
    transition: 'background 120ms ease',
    background: 'transparent',
  }),
  itemHover: {
    background: '#f8fafc',
  },
  message: {
    fontSize: 13,
    fontWeight: 500,
    color: '#0f172a',
    margin: 0,
    lineHeight: 1.35,
  },
  time: {
    fontSize: 11,
    color: '#94a3b8',
    marginTop: 2,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: '#2563eb',
    flexShrink: 0,
    marginTop: 5,
  },
  empty: {
    padding: '28px 16px',
    textAlign: 'center',
    color: '#94a3b8',
    fontSize: 13,
    lineHeight: 1.5,
  },
  footer: {
    borderTop: '1px solid #f1f5f9',
    padding: '10px 16px',
    textAlign: 'center',
    fontSize: 12,
    color: '#94a3b8',
  },
  settingsLink: {
    color: '#2563eb',
    textDecoration: 'none',
    fontWeight: 600,
  },
}

export default function NotificationBell({ unreadCount, onRead }) {
  const { user } = useAuth()
  const [open, setOpen] = useState(false)
  const [notifications, setNotifications] = useState([])
  const [loading, setLoading] = useState(false)
  const [hoveredId, setHoveredId] = useState(null)
  const closeTimer = useRef(null)
  const wrapperRef = useRef(null)

  const fetchNotifications = useCallback(async () => {
    if (!user) return
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/notifications/${user.id}`)
      if (res.ok) setNotifications(await res.json())
    } catch { /* offline */ }
    finally { setLoading(false) }
  }, [user])

  function handleMouseEnter() {
    clearTimeout(closeTimer.current)
    if (!open) {
      setOpen(true)
      fetchNotifications()
    }
  }

  function handleMouseLeave() {
    closeTimer.current = setTimeout(() => setOpen(false), 250)
  }

  async function markAllRead() {
    if (!user) return
    try {
      await fetch(`${API_BASE}/notifications/${user.id}/read`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mark_all: true }),
      })
      setNotifications(prev => prev.map(n => ({ ...n, read: true })))
      onRead?.()
    } catch { /* offline */ }
  }

  async function markOneRead(id) {
    if (!user) return
    try {
      await fetch(`${API_BASE}/notifications/${user.id}/read`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notification_ids: [id] }),
      })
      setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n))
      onRead?.()
    } catch { /* offline */ }
  }

  const unread = notifications.filter(n => !n.read).length

  return (
    <div
      ref={wrapperRef}
      style={popupStyles.wrapper}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Bell icon button */}
      <button
        className="nav-link nav-button"
        style={{ position: 'relative', display: 'flex', alignItems: 'center', padding: '8px 12px', background: 'transparent', cursor: 'pointer' }}
        aria-label="Notifications"
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
          <path d="M13.73 21a2 2 0 0 1-3.46 0" />
        </svg>
        {unreadCount > 0 && (
          <span style={{
            position: 'absolute',
            top: 2,
            right: 2,
            background: '#ef4444',
            color: '#fff',
            fontSize: 9,
            fontWeight: 700,
            borderRadius: '50%',
            minWidth: 16,
            height: 16,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            lineHeight: 1,
            padding: '0 3px',
          }}>
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {/* Dropdown popup */}
      {open && (
        <div style={popupStyles.dropdown}>
          <div style={popupStyles.header}>
            <p style={popupStyles.title}>Notifications</p>
            {unread > 0 && (
              <button style={popupStyles.markAll} onClick={markAllRead}>
                Mark all read
              </button>
            )}
          </div>

          <div style={popupStyles.list}>
            {loading && notifications.length === 0 ? (
              <div style={popupStyles.empty}>Loading…</div>
            ) : notifications.length === 0 ? (
              <div style={popupStyles.empty}>
                No notifications yet.<br />
                When someone likes a lot you saved or uploads a photo, it will appear here.
              </div>
            ) : (
              notifications.slice(0, 20).map(n => (
                <div
                  key={n.id}
                  style={{
                    ...popupStyles.item(n.read),
                    ...(hoveredId === n.id ? popupStyles.itemHover : {}),
                  }}
                  onClick={() => !n.read && markOneRead(n.id)}
                  onMouseEnter={() => setHoveredId(n.id)}
                  onMouseLeave={() => setHoveredId(null)}
                >
                  {n.type === 'like' && (
                    <span style={{ color: '#ef4444', fontSize: 16, flexShrink: 0, marginTop: 1 }}>♥</span>
                  )}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={popupStyles.message}>{n.message}</p>
                    <div style={popupStyles.time}>{timeAgo(n.created_at)}</div>
                  </div>
                  {!n.read && <span style={popupStyles.dot} />}
                </div>
              ))
            )}
          </div>

          <div style={popupStyles.footer}>
            <Link to="/profile" style={popupStyles.settingsLink} onClick={() => setOpen(false)}>
              Turn off in Settings
            </Link>
          </div>
        </div>
      )}
    </div>
  )
}
