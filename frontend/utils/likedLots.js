const API_BASE = 'http://localhost:8000'

export const getParcelKey = (parcel) => {
  if (!parcel) return ''
  return (
    parcel.parcel_number ||
    parcel.parcelNumber ||
    parcel.opa_id ||
    parcel.address ||
    `${parcel.lat ?? ''},${parcel.lon ?? ''}`
  )
}

const notifyUpdate = () => {
  window.dispatchEvent(new Event('likedLotsUpdated'))
}

export const fetchLikedLots = async (userId) => {
  if (!userId) return []
  const res = await fetch(`${API_BASE}/liked_lots?user_id=${encodeURIComponent(userId)}`)
  if (!res.ok) throw new Error('Failed to load liked lots')
  const data = await res.json()
  return Array.isArray(data.items) ? data.items : []
}

export const fetchLikeCount = async (parcelKey) => {
  if (!parcelKey) return 0
  const res = await fetch(`${API_BASE}/liked_lots/count?parcel_key=${encodeURIComponent(parcelKey)}`)
  if (!res.ok) return 0
  const data = await res.json()
  return Number(data.total) || 0
}

export const fetchLikeStatus = async (userId, parcelKey) => {
  if (!userId || !parcelKey) return false
  const res = await fetch(`${API_BASE}/liked_lots/status?user_id=${encodeURIComponent(userId)}&parcel_key=${encodeURIComponent(parcelKey)}`)
  if (!res.ok) return false
  const data = await res.json()
  return Boolean(data.liked)
}

export const toggleLike = async (userId, parcel) => {
  const key = getParcelKey(parcel)
  if (!userId || !key) return { liked: false, total_likes: 0 }
  const res = await fetch(`${API_BASE}/liked_lots/toggle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, parcel_key: key, parcel }),
  })
  if (!res.ok) throw new Error('Failed to toggle like')
  const data = await res.json()
  notifyUpdate()
  return data
}

export const removeLikeByKey = async (userId, parcelKey) => {
  if (!userId || !parcelKey) return
  const res = await fetch(`${API_BASE}/liked_lots/toggle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, parcel_key: parcelKey, parcel: {} }),
  })
  if (res.ok) notifyUpdate()
}
