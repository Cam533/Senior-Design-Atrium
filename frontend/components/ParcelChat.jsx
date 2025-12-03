import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

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
  const inputRef = useRef(null)
  const messagesContainerRef = useRef(null)
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true)
  const [expandedMessageIds, setExpandedMessageIds] = useState(new Set())

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
    // reset messages when opening parcel chat (ephemeral)
    setMessages([])
    setInputValue('')
    setScores(null)
    setActiveTab('summary')
    setCensusData(null)
    setTimeout(() => inputRef.current?.focus(), 200)
    
    if (parcel.lat && parcel.lon) {
      setLoadingScores(true)
      setLoadingCensus(true)
      
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
  }, [parcel])

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
        <div style={{ flex: 1, overflowY: 'auto', background: '#fbfbfb', padding: 12 }}>
          <div style={{ fontWeight: 800, marginBottom: 8, fontSize: 18, letterSpacing: 0.3, fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial", color: '#0f172a' }}>{parcel.address || 'Selected Parcel'}</div>
          
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
              const skipKeys = new Set(['address', 'lniaddresskey','build_rank','objectid','opa_id','shape__area','shape__length','lat','lon', 'date_update', 'owner2'])
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

          {/* Collapsible Census inside the same button so the button grows when expanded */}
          {censusData && (
            <div style={{ marginTop: 12 }}>
              <button
                onClick={() => setCensusExpanded(!censusExpanded)}
                aria-expanded={censusExpanded}
                style={{
                  width: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'stretch',
                  borderRadius: 8,
                  border: '1px solid #e6edf3',
                  background: 'white',
                  overflow: 'hidden',
                  cursor: 'pointer',
                  padding: 0,
                  transition: 'box-shadow 200ms ease, transform 200ms ease'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px' }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 14 }}>Census Data</div>
                  </div>
                  <div style={{ fontSize: 18, color: '#64748b', transform: censusExpanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 200ms ease' }}>{'▸'}</div>
                </div>

                <div style={{
                  maxHeight: censusExpanded ? 260 : 0,
                  overflow: 'hidden',
                  transition: 'max-height 300ms ease, opacity 180ms ease',
                  opacity: censusExpanded ? 1 : 0,
                }}>
                  <div style={{ padding: '10px 12px 14px 12px', display: 'flex', flexDirection: 'column', gap: 8, fontSize: 13 }}>
                    {censusData.category_code_description && (
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                        <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 140, textAlign: 'left' }}>Property Type:</div>
                        <div style={{ fontWeight: 500, textAlign: 'left' }}>{censusData.category_code_description}</div>
                      </div>
                    )}
                    {/* {censusData.tract_total_pop !== null && censusData.tract_total_pop !== undefined && (
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                        <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 140, textAlign: 'left' }}>Tract Population:</div>
                        <div style={{ fontWeight: 500, textAlign: 'left' }}>{Math.round(censusData.tract_total_pop).toLocaleString()}</div>
                      </div>
                    )} */}
                    {censusData.tract_median_income !== null && censusData.tract_median_income !== undefined && (
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                        <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 140, textAlign: 'left' }}>Median Income:</div>
                        <div style={{ fontWeight: 500, textAlign: 'left' }}>${Math.round(censusData.tract_median_income).toLocaleString()}</div>
                      </div>
                    )}
                    {censusData.tract_median_age !== null && censusData.tract_median_age !== undefined && (
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                        <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 140, textAlign: 'left' }}>Median Age:</div>
                        <div style={{ fontWeight: 500, textAlign: 'left' }}>{censusData.tract_median_age.toFixed(1)} years</div>
                      </div>
                    )}
                    {censusData.tract_median_home_value !== null && censusData.tract_median_home_value !== undefined && (
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                        <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 140, textAlign: 'left' }}>Median Home Value:</div>
                        <div style={{ fontWeight: 500, textAlign: 'left' }}>${Math.round(censusData.tract_median_home_value).toLocaleString()}</div>
                      </div>
                    )}
                    {censusData.tract_median_rent !== null && censusData.tract_median_rent !== undefined && (
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                        <div style={{ fontSize: 13, color: '#475569', fontWeight: 700, minWidth: 140, textAlign: 'left' }}>Median Rent:</div>
                        <div style={{ fontWeight: 500, textAlign: 'left' }}>${Math.round(censusData.tract_median_rent).toLocaleString()}/mo</div>
                      </div>
                    )}
                  </div>
                </div>
              </button>
            </div>
          )}

          {/* Geographic scores */}
          {scores ? (
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #eef2f7' }}>
              <div style={{ fontWeight: 700, marginBottom: 10, fontSize: 14 }}>Geographic Scores</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {['environmental','recreational','transit','walkability'].map((k) => {
                  const keyName = k + '_score'
                  const raw = scores?.[keyName]
                  const score = Number.isFinite(Number(raw)) ? Number(raw) : null
                  const display = score !== null ? score.toFixed(1) : 'N/A'
                  let bg = '#e2e8f0'
                  let color = '#0f172a'
                  if (score !== null) {
                    if (score >= 7) { bg = '#dcfce7'; color = '#166534' }
                    else if (score >= 4) { bg = '#fef3c7'; color = '#92400e' }
                    else { bg = '#fee2e2'; color = '#991b1b' }
                  }
                  return (
                    <div key={k} style={{ minWidth: 140, flex: '0 0 auto', borderRadius: 8, padding: '8px 10px', background: '#fff', boxShadow: '0 1px 2px rgba(0,0,0,0.04)', display: 'flex', flexDirection: 'column', gap: 8 }}>
                      <div style={{ fontSize: 12, color: '#64748b', textTransform: 'capitalize' }}>{k.replace('_',' ')}</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ fontWeight: 700, fontSize: 16 }}>{display}</div>
                        <div style={{ color: '#64748b', fontSize: 12 }}>/10</div>
                      </div>
                      <div style={{ height: 8, borderRadius: 6, background: '#f1f5f9', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: score !== null ? Math.min(100, Math.max(0, (score/10)*100)) + '%' : '0%', background: bg, transition: 'width 400ms ease' }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ) : (
            loadingScores && (
              <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #eef2f7', fontSize: 13, color: '#64748b' }}>
                Loading scores...
              </div>
            )
          )}
        </div>
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
