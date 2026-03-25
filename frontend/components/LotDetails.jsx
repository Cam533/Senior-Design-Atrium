// Full-window detailed lot information page
// Displayed when user clicks "View More" from the parcel summary
import { useState, useEffect } from 'react'
import '../styles/LotDetails.css'
import PlotImageGallery from './PlotImageGallery'
import { fetchLikeCount, fetchLikeStatus, toggleLike } from '../utils/likedLots'
import { useAuth } from '../src/context/AuthContext'

export default function LotDetails({ parcel = null, onBack = () => {}, scores: initialScores = null, loadingScores: initialLoadingScores = false, censusData: initialCensusData = null, loadingCensus: initialLoadingCensus = false }) {
  const [scores, setScores] = useState(initialScores)
  const [loadingScores, setLoadingScores] = useState(initialScores)
  const [censusData, setCensusData] = useState(initialCensusData)
  const [loadingCensus, setLoadingCensus] = useState(initialLoadingCensus)
  const [liked, setLiked] = useState(false)
  const [likeCount, setLikeCount] = useState(null)
  const [likeLoading, setLikeLoading] = useState(false)
  const { user } = useAuth()

  const scoreMeta = {
    environmental: {
      label: 'Environmental',
      description:
        'This score reflects environmental quality around the parcel, including nearby green cover and natural features. Higher scores indicate a cleaner, healthier surrounding environment with more tree cover and garden access.',
    },
    recreational: {
      label: 'Recreational',
      description:
        'This score measures access to parks and recreational sites nearby. Higher scores mean more and closer recreation opportunities within short walking distance.',
    },
    transit: {
      label: 'Transit',
      description:
        'This score captures proximity to public transportation options such as buses, trolleys, or rail. Higher scores indicate more transit stops within a short walk.',
    },
    walkability: {
      label: 'Walkability',
      description:
        'This score estimates how walk‑friendly the area is, based on nearby destinations like parks and gardens plus transit access. Higher scores mean more daily needs can be reached on foot.',
    },
  }

  const getScoreCategory = (score) => {
    if (score === null || score === undefined || Number.isNaN(Number(score))) return 'Not available'
    if (score >= 7) return 'High'
    if (score >= 4) return 'Moderate'
    return 'Low'
  }

  const getScoreMeaning = (score) => {
    if (score === null || score === undefined || Number.isNaN(Number(score))) {
      return 'This score is not available for the selected location.'
    }
    const category = getScoreCategory(score)
    return `${score.toFixed(1)} / 10 is considered ${category.toLowerCase()} for this category. Higher values indicate stronger access or better conditions.`
  }

  useEffect(() => {
    if (parcel && user?.id) {
      console.log('User and parcel found, fetching id')
      console.log('Parcel:', parcel)
      const image_key = parcel.objectid
      const key = parcel?.parcel_number || parcel?.parcelNumber || parcel?.opa_id || parcel?.address || `${parcel?.lat ?? ''},${parcel?.lon ?? ''}`
      if (key) {
        setLikeLoading(true)
        Promise.all([
          fetchLikeStatus(user.id, key),
          fetchLikeCount(key),
        ])
          .then(([status, count]) => {
            setLiked(status)
            setLikeCount(count)
          })
          .catch(() => {
            setLiked(false)
            setLikeCount(null)
          })
          .finally(() => setLikeLoading(false))
      }
    } else {
      console.log('No user or parcel, setting liked to false and like count to null')
      setLiked(false)
      setLikeCount(null)
    }
    // If scores were passed as props (from ParcelChat), use them directly
    if (initialScores !== null) {
      setScores(initialScores)
      return
    }
    
    if (!parcel || !parcel.lat || !parcel.lon) return
    
    setLoadingScores(true)
    fetch('http://localhost:8000/geographic_scores', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat: parcel.lat, lon: parcel.lon })
    })
      .then(r => r.json())
      .then(data => {
        setScores(data)
        setLoadingScores(false)
      })
      .catch(err => {
        console.error('Failed to fetch scores:', err)
        setLoadingScores(false)
      })
  }, [parcel, initialScores, user?.id])

  useEffect(() => {
    // If census data was passed as props, use it directly
    if (initialCensusData !== null) {
      setCensusData(initialCensusData)
      return
    }

    if (!parcel || !parcel.lat || !parcel.lon) return

    setLoadingCensus(true)
    fetch('http://localhost:8000/parcel_census_data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat: parcel.lat, lon: parcel.lon, radius_m: 100 })
    })
      .then(r => r.json())
      .then(data => {
        if (data.data) {
          setCensusData(data.data)
        }
        setLoadingCensus(false)
      })
      .catch(err => {
        console.error('Failed to fetch census data:', err)
        setLoadingCensus(false)
      })
  }, [parcel, initialCensusData])

  const handleToggleLike = async () => {
    if (!user?.id) return
    try {
      const result = await toggleLike(user.id, parcel)
      setLiked(result.liked)
      setLikeCount(result.total_likes)
    } catch (e) {
      // ignore
    }
  }

  if (!parcel) return null

  return (
    <div className="lot-details-page">
      {/* Header with back button */}
      <div className="lot-details-header">
        <button
          onClick={onBack}
          aria-label="Back to map"
          className="lot-details-back-button"
        >
          ←
        </button>
        <h1 className="lot-details-title">
          {parcel.address || 'Lot Details'}
        </h1>
        <button
          onClick={handleToggleLike}
          aria-label={liked ? 'Unlike this lot' : 'Like this lot'}
          className="lot-details-like-button"
          disabled={!user?.id}
          title={user?.id ? (liked ? 'Unlike this lot' : 'Like this lot') : 'Log in to like lots'}
        >
          {liked ? '♥' : '♡'}
        </button>
      </div>
      {(likeCount !== null || !user?.id) && (
        <div style={{ padding: '0 24px 12px 24px', color: '#64748b', fontSize: 12 }}>
          {!user?.id && 'Log in to save lots to your profile.'}
          {user?.id && likeCount !== null && (
            <span>
              {likeLoading ? 'Loading likes…' : `Liked by ${Math.max(0, likeCount - (liked ? 1 : 0))} other users`}
            </span>
          )}
        </div>
      )}

      {/* Main content area with sidebar layout */}
      <div className="lot-details-main">
        {/* Left sidebar for images/street view */}
        <div className="lot-details-sidebar">
          <PlotImageGallery parcelNumber={parcel?.objectid} />
        </div>

        {/* Right content area */}
        <div className="lot-details-content">
          {/* Placeholder sections for future content */}
          <div className="lot-details-section">
            <h2>Lot Information</h2>
            
            {censusData && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 16 }}>
                {censusData.category_code_description && (
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                    <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 120 }}>Property Type:</div>
                    <div style={{ fontWeight: 500, fontSize: 13 }}>{censusData.category_code_description}</div>
                  </div>
                )}
                {censusData.tract_median_income !== null && censusData.tract_median_income !== undefined && (
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                    <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 120 }}>Median Income:</div>
                    <div style={{ fontWeight: 500, fontSize: 13 }}>${Math.round(censusData.tract_median_income).toLocaleString()}</div>
                  </div>
                )}
                {censusData.tract_median_age !== null && censusData.tract_median_age !== undefined && (
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                    <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 120 }}>Median Age:</div>
                    <div style={{ fontWeight: 500, fontSize: 13 }}>{censusData.tract_median_age.toFixed(1)} years</div>
                  </div>
                )}
                {censusData.tract_median_home_value !== null && censusData.tract_median_home_value !== undefined && (
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                    <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 120 }}>Median Home Value:</div>
                    <div style={{ fontWeight: 500, fontSize: 13 }}>${Math.round(censusData.tract_median_home_value).toLocaleString()}</div>
                  </div>
                )}
                {censusData.tract_median_rent !== null && censusData.tract_median_rent !== undefined && (
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
                    <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 120 }}>Median Rent:</div>
                    <div style={{ fontWeight: 500, fontSize: 13 }}>${Math.round(censusData.tract_median_rent).toLocaleString()}/mo</div>
                  </div>
                )}
              </div>
            )}
            
            {loadingCensus && !censusData && (
              <p style={{ color: '#64748b', marginBottom: 16 }}>Loading census data...</p>
            )}
            
            <p>More detailed lot information will be displayed here.</p>
          </div>

          {/* Geographic Scores */}
          {scores ? (
            <div className="lot-details-section">
              <h2>Geographic Scores</h2>
              <div className="scores-grid">
                {['environmental','recreational','transit','walkability'].map((k) => {
                  const keyName = k + '_score'
                  const raw = scores?.[keyName]
                  const score = Number.isFinite(Number(raw)) ? Number(raw) : null
                  const display = score !== null ? score.toFixed(1) : 'N/A'
                  const meta = scoreMeta[k]
                  let bg = '#e2e8f0'
                  let color = '#0f172a'
                  if (score !== null) {
                    if (score >= 7) { bg = '#dcfce7'; color = '#166534' }
                    else if (score >= 4) { bg = '#fef3c7'; color = '#92400e' }
                    else { bg = '#fee2e2'; color = '#991b1b' }
                  }
                  return (
                    <div key={k} className="score-card" style={{ borderRadius: 10, padding: '14px', background: '#fff', boxShadow: '0 1px 3px rgba(0,0,0,0.06)', display: 'flex', flexDirection: 'column', gap: 10, boxSizing: 'border-box', border: '1px solid #e6edf3' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ fontSize: 13, color: '#64748b', fontWeight: 600 }}>{meta?.label ?? k}</div>
                        <div style={{
                          fontSize: 11,
                          fontWeight: 700,
                          color,
                          background: score === null ? '#e2e8f0' : bg,
                          padding: '2px 8px',
                          borderRadius: 999
                        }}>
                          {getScoreCategory(score)}
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                        <div style={{ fontWeight: 800, fontSize: 22, color }}>{display}</div>
                        <div style={{ color: '#94a3b8', fontSize: 12 }}>/10</div>
                      </div>
                      <div style={{ height: 10, borderRadius: 999, background: '#f1f5f9', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: score !== null ? Math.min(100, Math.max(0, (score / 10) * 100)) + '%' : '0%',
                          background: score === null ? '#e2e8f0' : bg,
                          transition: 'width 400ms ease'
                        }} />
                      </div>
                      <details style={{ fontSize: 12, color: '#334155', width: '100%', maxWidth: '100%', paddingRight: 8, paddingLeft: 2 }}>
                        <summary style={{ cursor: 'pointer', color: '#0f172a', fontWeight: 600 }}>What does this score mean?</summary>
                        <div style={{ marginTop: 8, lineHeight: 1.4, overflowWrap: 'anywhere', wordBreak: 'break-word', paddingRight: 10, boxSizing: 'border-box', maxWidth: '100%' }}>
                          <div style={{ marginBottom: 6 }}>{meta?.description}</div>
                          <div style={{ color: '#475569' }}>{getScoreMeaning(score)}</div>
                        </div>
                      </details>
                    </div>
                  )
                })}
              </div>

              {/* Distances: nearest park & nearest transit with address and pin */}
              <div className="lot-details-distances">
                <h3>Distances</h3>
                <div className="lot-details-distance-cards">
                  {scores.nearest_park && typeof scores.nearest_park === 'object' && scores.nearest_park.lat != null && scores.nearest_park.lon != null ? (
                    <div className="lot-details-distance-card">
                      <div className="lot-details-distance-card-header">
                        <a
                          href={`https://www.openstreetmap.org/?mlat=${scores.nearest_park.lat}&mlon=${scores.nearest_park.lon}&zoom=17`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="lot-details-distance-pin park"
                          title="View on map"
                          aria-label="View nearest park on map"
                        >
                          📍
                        </a>
                        <span className="lot-details-distance-label">Nearest park</span>
                        <span className="lot-details-distance-value">
                          {scores.nearest_park.distance_m >= 1000
                            ? `${(scores.nearest_park.distance_m / 1000).toFixed(2)} km`
                            : `${Math.round(scores.nearest_park.distance_m)} m`}
                        </span>
                      </div>
                      {(scores.nearest_park.address || scores.nearest_park.name) ? (
                        <div className="lot-details-distance-address">
                          {scores.nearest_park.name && scores.nearest_park.address && scores.nearest_park.name !== scores.nearest_park.address
                            ? <><span className="lot-details-distance-address-label">Name:</span> {scores.nearest_park.name}<br /><span className="lot-details-distance-address-label">Address:</span> {scores.nearest_park.address}</>
                            : <><span className="lot-details-distance-address-label">Address:</span> {scores.nearest_park.address || scores.nearest_park.name}</>}
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div className="lot-details-distance-card">
                      <div className="lot-details-distance-card-header">
                        <span className="lot-details-distance-pin park" style={{ cursor: 'default' }}>📍</span>
                        <span className="lot-details-distance-label">Nearest park</span>
                        <span className="lot-details-distance-value">
                          {scores.distance_to_nearest_park_m != null
                            ? (scores.distance_to_nearest_park_m >= 1000 ? `${(scores.distance_to_nearest_park_m / 1000).toFixed(2)} km` : `${Math.round(scores.distance_to_nearest_park_m)} m`)
                            : 'N/A'}
                        </span>
                      </div>
                    </div>
                  )}
                  {scores.nearest_transit_stop && typeof scores.nearest_transit_stop === 'object' && scores.nearest_transit_stop.lat != null && scores.nearest_transit_stop.lon != null ? (
                    <div className="lot-details-distance-card">
                      <div className="lot-details-distance-card-header">
                        <a
                          href={`https://www.openstreetmap.org/?mlat=${scores.nearest_transit_stop.lat}&mlon=${scores.nearest_transit_stop.lon}&zoom=17`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="lot-details-distance-pin transit"
                          title="View on map"
                          aria-label="View nearest transit stop on map"
                        >
                          🚏
                        </a>
                        <span className="lot-details-distance-label">Nearest public transit stop</span>
                        <span className="lot-details-distance-value">
                          {scores.nearest_transit_stop.distance_m >= 1000
                            ? `${(scores.nearest_transit_stop.distance_m / 1000).toFixed(2)} km`
                            : `${Math.round(scores.nearest_transit_stop.distance_m)} m`}
                        </span>
                      </div>
                      {(scores.nearest_transit_stop.address || scores.nearest_transit_stop.name) ? (
                        <div className="lot-details-distance-address">
                          {scores.nearest_transit_stop.name && scores.nearest_transit_stop.address && scores.nearest_transit_stop.name !== scores.nearest_transit_stop.address
                            ? <><span className="lot-details-distance-address-label">Name:</span> {scores.nearest_transit_stop.name}<br /><span className="lot-details-distance-address-label">Address:</span> {scores.nearest_transit_stop.address}</>
                            : <><span className="lot-details-distance-address-label">Address:</span> {scores.nearest_transit_stop.address || scores.nearest_transit_stop.name}</>}
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div className="lot-details-distance-card">
                      <div className="lot-details-distance-card-header">
                        <span className="lot-details-distance-pin transit" style={{ cursor: 'default' }}>🚏</span>
                        <span className="lot-details-distance-label">Nearest public transit stop</span>
                        <span className="lot-details-distance-value">
                          {scores.distance_to_nearest_transit_stop_m != null
                            ? (scores.distance_to_nearest_transit_stop_m >= 1000 ? `${(scores.distance_to_nearest_transit_stop_m / 1000).toFixed(2)} km` : `${Math.round(scores.distance_to_nearest_transit_stop_m)} m`)
                            : 'N/A'}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            loadingScores && (
              <div className="lot-details-section">
                <p>Loading scores...</p>
              </div>
            )
          )}

          <div className="lot-details-section">
            <h2>Development Potential</h2>
            <p>Development potential and recommendations will be shown here.</p>
          </div>

          <div className="lot-details-section">
            <h2>Additional Resources</h2>
            <p>Additional resources and links will be available here.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
