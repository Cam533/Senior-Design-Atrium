import { useEffect, useState } from 'react'
import LotDetails from './LotDetails'
import { fetchLikedLots, removeLikeByKey } from '../utils/likedLots'
import { useAuth } from '../src/context/AuthContext'

export default function LikedLots() {
  const [likedLots, setLikedLots] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const { user } = useAuth()

  useEffect(() => {
    if (!user?.id) {
      setLikedLots([])
      return
    }
    const handleUpdate = async () => {
      try {
        setLoading(true)
        setError(null)
        const items = await fetchLikedLots(user.id)
        setLikedLots(items)
      } catch (e) {
        setError('Failed to load liked lots.')
      } finally {
        setLoading(false)
      }
    }
    handleUpdate()
    window.addEventListener('likedLotsUpdated', handleUpdate)
    return () => window.removeEventListener('likedLotsUpdated', handleUpdate)
  }, [user?.id])

  if (selected) {
    return (
      <LotDetails
        parcel={selected.parcel}
        onBack={() => setSelected(null)}
      />
    )
  }

  return (
    <div style={{ padding: 16, maxWidth: 900, margin: '0 auto', width: '100%' }}>
      <div style={{ fontSize: 20, fontWeight: 800, marginBottom: 12, color: '#0f172a' }}>
        Liked Lots
      </div>
      {!user?.id ? (
        <div style={{ color: '#64748b', fontSize: 14 }}>
          Log in to view your liked lots.
        </div>
      ) : loading ? (
        <div style={{ color: '#64748b', fontSize: 14 }}>Loading liked lots…</div>
      ) : error ? (
        <div style={{ color: '#b91c1c', fontSize: 14 }}>{error}</div>
      ) : likedLots.length === 0 ? (
        <div style={{ color: '#64748b', fontSize: 14 }}>
          No liked lots yet. Click the heart on a lot to save it here.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {likedLots.map((lot) => {
            const parcel = typeof lot.parcel === 'string' ? (() => { try { return JSON.parse(lot.parcel) } catch (e) { return {} } })() : (lot.parcel || {})
            return (
            <div
              key={lot.parcel_key || lot.key}
              style={{
                border: '1px solid #e6edf3',
                borderRadius: 10,
                padding: '12px 14px',
                background: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
                boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
              }}
            >
              <div>
                <div style={{ fontWeight: 700, color: '#0f172a' }}>
                  {parcel?.address || 'Saved Lot'}
                </div>
                {parcel?.zoningbasedistrict && (
                  <div style={{ fontSize: 12, color: '#64748b' }}>
                    Zoning: {parcel.zoningbasedistrict}
                  </div>
                )}
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  type="button"
                  onClick={() => setSelected({ ...lot, parcel })}
                  style={{
                    border: '1px solid #0f172a',
                    background: '#0f172a',
                    color: 'white',
                    padding: '8px 12px',
                    borderRadius: 8,
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                >
                  View Details
                </button>
                <button
                  type="button"
                  onClick={() => removeLikeByKey(user.id, lot.parcel_key || lot.key)}
                  aria-label="Remove from liked lots"
                  style={{
                    border: '1px solid #e2e8f0',
                    background: 'white',
                    color: '#b91c1c',
                    width: 34,
                    height: 34,
                    borderRadius: 999,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 16,
                  }}
                >
                  ♥
                </button>
              </div>
            </div>
          )})}
        </div>
      )}
    </div>
  )
}
