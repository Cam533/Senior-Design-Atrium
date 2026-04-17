import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import LotDetails from './LotDetails'
import { fetchLikeCount, fetchLikeStatus, toggleLike } from '../utils/likedLots'
import { useAuth } from '../src/context/AuthContext'

const API_URL = 'http://localhost:8000/chat'

// Independent ephemeral parcel chat: own messages, input, and send behavior.
// No global state is modified; closing and reopening resets the panel.
export default function ParcelChat({ parcel = null, onClose = () => {} }) {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [scores, setScores] = useState(null)
  const [censusData, setCensusData] = useState(null)
  const [loadingCensus, setLoadingCensus] = useState(false)
  const [loadingScores, setLoadingScores] = useState(false)
  const [censusExpanded, setCensusExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState('summary')
  const [hoveredTab, setHoveredTab] = useState(null)
  const [showRecommendations, setShowRecommendations] = useState(true)
  const [showDetailsPage, setShowDetailsPage] = useState(false)
  const [liked, setLiked] = useState(false)
  const [likeCount, setLikeCount] = useState(null)
  const [likeLoading, setLikeLoading] = useState(false)
  const { user } = useAuth()
  const inputRef = useRef(null)
  const messagesContainerRef = useRef(null)
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true)
  const [expandedMessageIds, setExpandedMessageIds] = useState(new Set())

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

  const toggleMessageExpanded = (id) => {
    setExpandedMessageIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  useEffect(() => {
    if (!parcel) return
    if (user?.id) {
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
      setLiked(false)
      setLikeCount(null)
    }
    // reset messages when opening parcel chat (ephemeral)
    setMessages([])
    setInputValue('')
    setScores(null)
    setActiveTab('summary')
    setCensusData(null)
    setTimeout(() => inputRef.current?.focus(), 200)
    
    if (parcel.lat && parcel.lon) {
      setLoadingCensus(true)
      
      // Use precomputed scores already attached to the parcel (avoid recalculating).
      // Expected fields: environmental_score, recreational_score, transit_score, walkability_score
      const precomputedScores = {
        environmental_score: parcel.environmental_score,
        recreational_score: parcel.recreational_score,
        transit_score: parcel.transit_score,
        walkability_score: parcel.walkability_score,
      }

      if (precomputedScores) {
        setScores(precomputedScores)
        setLoadingScores(false)
      } else {
        setLoadingScores(true)
        // Fetch geographic scores
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
      }
      
      // Fetch census data (debug: log status and body)
      fetch('http://localhost:8000/parcel_census_data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat: parcel.lat, lon: parcel.lon, radius_m: 50 })
      })
        .then(async (r) => {
          const text = await r.text()
          let parsed = null
          try {
            parsed = JSON.parse(text)
          } catch (e) {
            console.log('parcel_census_data: response is not valid JSON', { status: r.status, ok: r.ok, text })
          }
          console.log('parcel_census_data response', { status: r.status, ok: r.ok, body: parsed ?? text })
          if (parsed && parsed.data) {
            setCensusData(parsed.data)
          }
          setLoadingCensus(false)
        })
        .catch(err => {
          console.error('Failed to fetch census data:', err)
          setLoadingCensus(false)
        })
    }
    setShowRecommendations(true)
  }, [parcel, user?.id])

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

  // Helper component to render a message bubble with optional truncation
  function MessageBubble({ msg, expanded, onToggle }) {
    const limit = 350
    const isLong = typeof msg.text === 'string' && msg.text.length > limit
    const displayText = isLong && !expanded ? msg.text.slice(0, limit) + '…' : msg.text

    const commonStyle = {
      display: 'inline-block',
      maxWidth: '86%',
      wordBreak: 'normal',
      overflowWrap: 'break-word',
      whiteSpace: 'pre-wrap',
      fontSize: 13,
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial",
      lineHeight: '1.3',
    }

    const isBot = msg.sender === 'bot'

    const bubbleStyle = isBot
      ? { background: '#f1f5f9', padding: 8, borderRadius: 8, ...commonStyle }
      : { background: '#0f172a', color: 'white', padding: 8, borderRadius: 8, textAlign: 'right', ...commonStyle, fontSize: 13 }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: isBot ? 'flex-start' : 'flex-end' }}>
        <div style={bubbleStyle}>
          {isBot ? (
            <ReactMarkdown
              components={{
                h1: ({node, ...props}) => <div style={{ fontSize: 16, fontWeight: 700 }} {...props} />, 
                h2: ({node, ...props}) => <div style={{ fontSize: 15, fontWeight: 700 }} {...props} />,
                h3: ({node, ...props}) => <div style={{ fontSize: 14, fontWeight: 700 }} {...props} />,
                p: ({node, ...props}) => <div style={{ margin: 0 }} {...props} />,
              }}
            >
              {displayText}
            </ReactMarkdown>
          ) : (
            displayText
          )}
        </div>
        {isLong && (
          <div style={{ marginTop: 6 }}>
            <button
              onClick={() => {
                onToggle && onToggle(msg.id)
                // if the user hasn't scrolled up, keep the view pinned to the bottom
                try {
                  const el = messagesContainerRef.current
                  if (el) {
                    // schedule a frame so layout updates before scrolling
                    window.requestAnimationFrame(() => {
                      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
                    })
                  }
                } catch (e) {
                  // ignore
                }
              }}
              style={{ background: 'transparent', border: 'none', color: isBot ? '#2563eb' : '#93c5fd', cursor: 'pointer', padding: 0 }}
            >
              {expanded ? 'Show less' : 'Read more'}
            </button>
          </div>
        )}
      </div>
    )
  }

  const callChat = async (message) => {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    })
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    return response.json()
  }

  const handleSend = async (e) => {
    e?.preventDefault()
    if (!inputValue.trim() || isLoading) return

    const userMessage = { id: Date.now(), text: inputValue, sender: 'user' }
    setMessages(prev => [...prev, userMessage])
    setShowRecommendations(false)
    const question = inputValue
    setInputValue('')
    setIsLoading(true)

    const loadingMessage = { id: Date.now() + 1, text: 'Thinking...', sender: 'bot', isLoading: true }
    setMessages(prev => [...prev, loadingMessage])

    try {
      // Provide parcel context to backend by prefixing the message
      const parcelId = parcel.address ? `Parcel: ${parcel.address}` : 'Parcel'
      const fullMessage = `${parcelId}\nContext: ${JSON.stringify(parcel)}\nQuestion: ${question}`
      const data = await callChat(fullMessage)
      setMessages(prev => {
        const withoutLoading = prev.filter(m => !m.isLoading)
        return [...withoutLoading, { id: Date.now() + 2, text: data.message || "Sorry, I couldn't process that request.", sender: 'bot' }]
      })
    } catch (err) {
      console.error('ParcelChat error', err)
      setMessages(prev => {
        const withoutLoading = prev.filter(m => !m.isLoading)
        return [...withoutLoading, { id: Date.now() + 2, text: "Sorry, I'm having trouble connecting to the server.", sender: 'bot' }]
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Start a chat from a recommended prompt: inject user message, switch to chat tab, call backend
  const startRecommendedChat = async (question) => {
    if (!question || isLoading) return
    // hide the recommendation buttons immediately when a recommended topic is chosen
    setShowRecommendations(false)
    const userMessage = { id: Date.now(), text: question, sender: 'user' }
    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)
    // show loading bubble
    const loadingMessage = { id: Date.now() + 1, text: 'Thinking...', sender: 'bot', isLoading: true }
    setMessages(prev => [...prev, loadingMessage])
    setActiveTab('chat')
    try {
      const parcelId = parcel && parcel.address ? `Parcel: ${parcel.address}` : 'Parcel'
      const fullMessage = `${parcelId}\nContext: ${JSON.stringify(parcel)}\nQuestion: ${question}`
      const data = await callChat(fullMessage)
      setMessages(prev => {
        const withoutLoading = prev.filter(m => !m.isLoading)
        return [...withoutLoading, { id: Date.now() + 2, text: data.message || "Sorry, I couldn't process that request.", sender: 'bot' }]
      })
    } catch (err) {
      console.error('Recommended chat error', err)
      setMessages(prev => {
        const withoutLoading = prev.filter(m => !m.isLoading)
        return [...withoutLoading, { id: Date.now() + 2, text: "Sorry, I'm having trouble connecting to the server.", sender: 'bot' }]
      })
    } finally {
      setIsLoading(false)
      setTimeout(() => inputRef.current?.focus(), 200)
    }
  }

  // Keep the messages view pinned to the bottom when new messages arrive,
  // unless the user has scrolled up (we toggle shouldAutoScroll in the onScroll handler).
  useEffect(() => {
    const el = messagesContainerRef.current
    if (!el) return
    if (shouldAutoScroll) {
      // small timeout to ensure the new message is rendered
      window.requestAnimationFrame(() => {
        el.scrollTo({ top: el.scrollHeight })
      })
    }
  }, [messages, shouldAutoScroll])

  if (!parcel) return null

  return (
    <div style={{
      position: 'absolute',
      right: 24,
      top: 100,
      width: 420,
      height: '75vh',
      background: 'white',
      borderRadius: 10,
      boxShadow: '0 8px 24px rgba(15, 23, 42, 0.12)',
      zIndex: 2000,
      overflow: 'hidden',
      overflowX: 'hidden',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Top-left close X */}
      <button
        onClick={() => onClose()}
        aria-label="Close parcel chat"
        style={{
          position: 'absolute',
          left: 12,
          top: 12,
          width: 32,
          height: 32,
          borderRadius: 6,
          border: 'none',
          background: 'transparent',
          color: '#0f172a',
          fontSize: 18,
          fontWeight: 700,
          cursor: 'pointer',
          zIndex: 2600,
        }}
      >
        ✕
      </button>
      {/* Tabs */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '12px', borderBottom: '1px solid #eef2f7' }}>
        <button
          onClick={() => setActiveTab('summary')}
          onMouseEnter={() => setHoveredTab('summary')}
          onMouseLeave={() => setHoveredTab(null)}
          style={{
            padding: '8px 12px',
            border: '1px solid #e6edf3',
            background: 'white',
            color: activeTab === 'summary' ? '#0f172a' : (hoveredTab === 'summary' ? '#0f172a' : '#475569'),
            cursor: 'pointer',
            fontWeight: 700,
            borderBottom: activeTab === 'summary' ? '3px solid #0f172a' : (hoveredTab === 'summary' ? '3px solid rgba(15,23,42,0.12)' : '1px solid #e6edf3'),
            borderRadius: 8,
            transition: 'color 140ms ease, border-bottom-color 140ms ease'
          }}
        >
          Summary
        </button>
        <button
          onClick={() => { setActiveTab('chat'); setTimeout(() => inputRef.current?.focus(), 120); }}
          onMouseEnter={() => setHoveredTab('chat')}
          onMouseLeave={() => setHoveredTab(null)}
          style={{
            padding: '8px 12px',
            border: '1px solid #e6edf3',
            background: 'white',
            color: activeTab === 'chat' ? '#0f172a' : (hoveredTab === 'chat' ? '#0f172a' : '#475569'),
            cursor: 'pointer',
            fontWeight: 700,
            borderBottom: activeTab === 'chat' ? '3px solid #0f172a' : (hoveredTab === 'chat' ? '3px solid rgba(15,23,42,0.12)' : '1px solid #e6edf3'),
            borderRadius: 8,
            transition: 'color 140ms ease, border-bottom-color 140ms ease'
          }}
        >
          Chat
        </button>
      </div>

      {/* Unified scrollable summary + scores + census area */}
      {activeTab === 'summary' && (
        <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', background: '#fbfbfb', padding: '12px 16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 4 }}>
            <div style={{ fontWeight: 800, fontSize: 18, letterSpacing: 0.3, fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial", color: '#0f172a' }}>
              {parcel.address || 'Selected Parcel'}
            </div>
            <button
              type="button"
              onClick={handleToggleLike}
              aria-label={liked ? 'Unlike this lot' : 'Like this lot'}
              style={{
                border: '1px solid #e2e8f0',
                background: liked ? '#fee2e2' : 'white',
                color: liked ? '#b91c1c' : '#0f172a',
                width: 34,
                height: 34,
                borderRadius: 999,
                cursor: user?.id ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 16,
                boxShadow: '0 1px 2px rgba(0,0,0,0.04)'
              }}
              disabled={!user?.id}
              title={user?.id ? (liked ? 'Unlike this lot' : 'Like this lot') : 'Log in to like lots'}
            >
              {liked ? '♥' : '♡'}
            </button>
          </div>
          {(likeCount !== null || !user?.id) && (
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
              {!user?.id && 'Log in to save lots to your profile.'}
              {user?.id && likeCount !== null && (
                <span>
                  {likeLoading ? 'Loading likes…' : `Liked by ${Math.max(0, likeCount - (liked ? 1 : 0))} other users`}
                </span>
              )}
            </div>
          )}
          
          <div style={{ fontSize: 13, color: '#111', display: 'flex', flexDirection: 'column', gap: 6 }}>
            {(() => {
              const props = parcel || {}
              const labelMap = {
                owner1: 'Owner',
                bldg_desc: 'Land Type',
                councildistrict: 'Council District',
                zoningbasedistrict: 'Zoning',
                zipcode: 'ZIP Code',
                land_rank: 'Vacancy Likelihood',
                date_update: 'Last Update'
              }
              const prettifyKey = (k) => String(k).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
              const formatValue = (k, v) => {
                if (v === null || v === undefined || String(v).trim() === '') return null
                if (k === 'Shape__Area' || k === 'Shape__Length' || k === 'land_rank') {
                  const num = Number(v)
                  if (Number.isFinite(num)) {
                    if (k === 'land_rank') return num.toFixed(2)
                    return num.toLocaleString(undefined, { maximumFractionDigits: 2 })
                  }
                }
                if (k === 'lat' || k === 'lon') {
                  const num = Number(v)
                  if (Number.isFinite(num)) {
                    return num.toFixed(6)
                  }
                }
                if (k === 'date_update') {
                  try { const d = new Date(v); if (!isNaN(d)) return d.toLocaleDateString() } catch (e) {}
                }
                return String(v)
              }

              const keysToShow = ['owner1','bldg_desc','zoningbasedistrict','councildistrict','zipcode','land_rank', 'date_update']
              const used = new Set()
              const rows = []
              for (const k of keysToShow) {
                if (props[k] !== undefined) {
                  const val = formatValue(k, props[k])
                  if (val !== null) {
                    rows.push(
                      <div key={k} style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                        <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 150 }}>
                          {labelMap[k] || prettifyKey(k)}:
                        </div>
                        <div style={{ fontSize: 14, color: '#0f172a', lineHeight: 1.3, wordBreak: 'break-word' }}>
                          {val}
                        </div>
                      </div>
                    )
                    used.add(k)
                  }
                }
              }
              let extraCount = 0
              const skipKeys = new Set([
                'address',
                'lniaddresskey',
                'build_rank',
                'objectid',
                'opa_id',
                'shape__area',
                'shape__length',
                'lat',
                'lon',
                'date_update',
                'owner2',
                // hide score fields from raw property rows; shown in score cards below
                'environmental_score',
                'recreational_score',
                'transit_score',
                'walkability_score',
                'distance_to_nearest_park_m',
                'distance_to_nearest_transit_stop_m',
                'nearest_park',
                'nearest_transit_stop',
              ])
              for (const [k,v] of Object.entries(props)) {
                if (used.has(k)) continue
                if (extraCount >= 6) break
                if (skipKeys.has(String(k).toLowerCase())) continue
                const val = formatValue(k, v)
                if (val !== null) {
                  const label = labelMap[k] || prettifyKey(k)
                  rows.push(<div key={k}><strong>{label}:</strong> {val}</div>)
                  extraCount += 1
                }
              }
              return rows
            })()}

            {loadingCensus && (
              <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #eee', fontSize: 13, color: '#64748b' }}>
                Loading data...
              </div>
            )}
          </div>

          {/* Geographic scores */}
          {scores ? (
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #eef2f7' }}>
              <div style={{ fontWeight: 700, marginBottom: 10, fontSize: 14 }}>Geographic Scores</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
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
                    <div key={k} style={{ borderRadius: 10, padding: '12px 14px', background: '#fff', boxShadow: '0 1px 3px rgba(0,0,0,0.06)', display: 'flex', flexDirection: 'column', gap: 8, boxSizing: 'border-box' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>{meta?.label ?? k}</div>
                        <div style={{
                          fontSize: 10,
                          fontWeight: 700,
                          color,
                          background: score === null ? '#e2e8f0' : bg,
                          padding: '2px 6px',
                          borderRadius: 999
                        }}>
                          {getScoreCategory(score)}
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                        <div style={{ fontWeight: 800, fontSize: 20, color }}>{display}</div>
                        <div style={{ color: '#94a3b8', fontSize: 11 }}>/10</div>
                      </div>
                      <div style={{ height: 8, borderRadius: 999, background: '#f1f5f9', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: score !== null ? Math.min(100, Math.max(0, (score / 10) * 100)) + '%' : '0%',
                          background: score === null ? '#e2e8f0' : bg,
                          transition: 'width 400ms ease'
                        }} />
                      </div>
                    </div>
                  )
                })}
              </div>
              {/* Distances: nearest park & transit with address and pin */}
              <div style={{ marginTop: 12 }}>
                <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 13, color: '#475569' }}>Distances</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {scores.nearest_park && typeof scores.nearest_park === 'object' && scores.nearest_park.lat != null ? (
                    <div style={{ padding: '10px 12px', background: '#fff', borderRadius: 10, border: '1px solid #e6edf3', boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <a href={`https://www.openstreetmap.org/?mlat=${scores.nearest_park.lat}&mlon=${scores.nearest_park.lon}&zoom=17`} target="_blank" rel="noopener noreferrer" style={{ width: 32, height: 32, borderRadius: 8, background: '#dcfce7', border: '1px solid #bbf7d0', color: '#166534', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, textDecoration: 'none' }} title="View on map">📍</a>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 12, color: '#475569', fontWeight: 600 }}>Nearest park</div>
                          {(scores.nearest_park.address || scores.nearest_park.name) && (
                            <div style={{ fontSize: 13, color: '#64748b', marginTop: 2 }}>
                              {scores.nearest_park.name && scores.nearest_park.address && scores.nearest_park.name !== scores.nearest_park.address
                                ? <>Name: {scores.nearest_park.name} · Address: {scores.nearest_park.address}</>
                                : <>Address: {scores.nearest_park.address || scores.nearest_park.name}</>}
                            </div>
                          )}
                        </div>
                        <span style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>
                          {scores.nearest_park.distance_m >= 1000 ? `${(scores.nearest_park.distance_m / 1000).toFixed(2)} km` : `${Math.round(scores.nearest_park.distance_m)} m`}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', background: '#f8fafc', borderRadius: 8, border: '1px solid #e6edf3', fontSize: 12 }}>
                      <span style={{ width: 32, height: 32, borderRadius: 8, background: '#e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>📍</span>
                      <span style={{ color: '#475569' }}>Nearest park</span>
                      <span style={{ fontWeight: 700, color: '#0f172a', marginLeft: 'auto' }}>
                        {scores.distance_to_nearest_park_m != null ? (scores.distance_to_nearest_park_m >= 1000 ? `${(scores.distance_to_nearest_park_m / 1000).toFixed(2)} km` : `${Math.round(scores.distance_to_nearest_park_m)} m`) : 'N/A'}
                      </span>
                    </div>
                  )}
                  {scores.nearest_transit_stop && typeof scores.nearest_transit_stop === 'object' && scores.nearest_transit_stop.lat != null ? (
                    <div style={{ padding: '10px 12px', background: '#fff', borderRadius: 10, border: '1px solid #e6edf3', boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <a href={`https://www.openstreetmap.org/?mlat=${scores.nearest_transit_stop.lat}&mlon=${scores.nearest_transit_stop.lon}&zoom=17`} target="_blank" rel="noopener noreferrer" style={{ width: 32, height: 32, borderRadius: 8, background: '#dbeafe', border: '1px solid #bfdbfe', color: '#1e40af', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, textDecoration: 'none' }} title="View on map">🚏</a>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 12, color: '#475569', fontWeight: 600 }}>Nearest transit stop</div>
                          {(scores.nearest_transit_stop.address || scores.nearest_transit_stop.name) && (
                            <div style={{ fontSize: 13, color: '#64748b', marginTop: 2 }}>
                              {scores.nearest_transit_stop.name && scores.nearest_transit_stop.address && scores.nearest_transit_stop.name !== scores.nearest_transit_stop.address
                                ? <>Name: {scores.nearest_transit_stop.name} · Address: {scores.nearest_transit_stop.address}</>
                                : <>Address: {scores.nearest_transit_stop.address || scores.nearest_transit_stop.name}</>}
                            </div>
                          )}
                        </div>
                        <span style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>
                          {scores.nearest_transit_stop.distance_m >= 1000 ? `${(scores.nearest_transit_stop.distance_m / 1000).toFixed(2)} km` : `${Math.round(scores.nearest_transit_stop.distance_m)} m`}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', background: '#f8fafc', borderRadius: 8, border: '1px solid #e6edf3', fontSize: 12 }}>
                      <span style={{ width: 32, height: 32, borderRadius: 8, background: '#e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>🚏</span>
                      <span style={{ color: '#475569' }}>Nearest transit stop</span>
                      <span style={{ fontWeight: 700, color: '#0f172a', marginLeft: 'auto' }}>
                        {scores.distance_to_nearest_transit_stop_m != null ? (scores.distance_to_nearest_transit_stop_m >= 1000 ? `${(scores.distance_to_nearest_transit_stop_m / 1000).toFixed(2)} km` : `${Math.round(scores.distance_to_nearest_transit_stop_m)} m`) : 'N/A'}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            loadingScores && (
              <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #eef2f7', fontSize: 13, color: '#64748b' }}>
                Loading scores...
              </div>
            )
          )}

          {/* View More button at bottom */}
          <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid #eef2f7' }}>
            <button
              onClick={() => setShowDetailsPage(true)}
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: 8,
                border: '1px solid #e6edf3',
                background: '#0f172a',
                color: 'white',
                fontWeight: 700,
                fontSize: 14,
                cursor: 'pointer',
                transition: 'background 140ms ease'
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = '#1e293b'}
              onMouseLeave={(e) => e.currentTarget.style.background = '#0f172a'}
            >
              View More Details
            </button>
          </div>
        </div>
      )}

      {/* Lot Details Page Modal */}
      {showDetailsPage && (
        <LotDetails parcel={parcel} onBack={() => setShowDetailsPage(false)} scores={scores} loadingScores={loadingScores} censusData={censusData} loadingCensus={loadingCensus} />
      )}

      {activeTab === 'chat' && (
        <>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: 12, background: '#fbfbfb', flex: 1, minHeight: 0 }}>
            {showRecommendations && (
              <>
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 12, color: '#52606d' }}>Recommended topics to ask about:</div>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button
                    className="chat-send-button recommendation-button"
                    onClick={() => startRecommendedChat('Is this property likely to be vacant or at risk of vacancy?')}
                    title="Is this property likely to be vacant or at risk of vacancy?"
                    style={{ padding: '6px 10px', borderRadius: 999, border: '1px solid #e6edf3', background: 'white', cursor: 'pointer', fontSize: 13, color: '#0f172a' }}
                  >
                    Vacancy likelihood
                  </button>
                  <button
                    className="chat-send-button recommendation-button"
                    onClick={() => startRecommendedChat('What nearby amenities (parks, transit, grocery) are within walking distance of this parcel?')}
                    title="What nearby amenities are within walking distance of this parcel?"
                    style={{ padding: '6px 10px', borderRadius: 999, border: '1px solid #e6edf3', background: 'white', cursor: 'pointer', fontSize: 13, color: '#0f172a' }}
                  >
                    Nearby amenities
                  </button>
                  <button
                    className="chat-send-button recommendation-button"
                    onClick={() => startRecommendedChat('What zoning restrictions or development constraints apply to this lot?')}
                    title="What zoning restrictions or development constraints apply to this lot?"
                    style={{ padding: '6px 10px', borderRadius: 999, border: '1px solid #e6edf3', background: 'white', cursor: 'pointer', fontSize: 13, color: '#0f172a' }}
                  >
                    Zoning constraints
                  </button>
                </div>
              </>
            )}

            <div
              ref={messagesContainerRef}
              onScroll={() => {
                const el = messagesContainerRef.current
                if (!el) return
                const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 64
                setShouldAutoScroll(nearBottom)
              }}
              style={{ paddingTop: 8, overflowY: 'auto', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', gap: 8, paddingBottom: 96 }}
            >
              {messages.map(msg => (
                <div key={msg.id} style={{ marginBottom: 10, display: 'flex', justifyContent: msg.sender === 'bot' ? 'flex-start' : 'flex-end' }}>
                  <MessageBubble msg={msg} expanded={expandedMessageIds.has(msg.id)} onToggle={toggleMessageExpanded} />
                </div>
              ))}
            </div>
          </div>

          <form
            onSubmit={handleSend}
            style={{
              display: 'flex',
              gap: 8,
              padding: 12,
              borderTop: 'none',
              background: 'rgba(251,251,251,0.98)',
              position: 'absolute',
              left: 12,
              right: 12,
              bottom: 12,
              zIndex: 2400,
              borderRadius: 8,
              boxShadow: '0 -6px 18px rgba(15, 23, 42, 0.06)',
            }}
          >
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about this lot..."
              style={{ flex: 1, padding: '8px 8px 8px 8px', borderRadius: 8, border: '1px solid #e2e8f0' }}
            />
            <button type="submit" className="chat-send-button">Send</button>
          </form>
        </>
      )}
    </div>
  )
}
