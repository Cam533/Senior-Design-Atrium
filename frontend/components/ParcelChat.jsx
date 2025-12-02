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
  const [loadingScores, setLoadingScores] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    if (!parcel) return
    // reset messages when opening parcel chat (ephemeral)
    setMessages([])
    setInputValue('')
    setScores(null)
    setCensusData(null)
    setTimeout(() => inputRef.current?.focus(), 200)
    
    if (parcel.lat && parcel.lon) {
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
      
      // Fetch census data
      fetch('http://localhost:8000/parcel_census_data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat: parcel.lat, lon: parcel.lon, radius_m: 50 })
      })
        .then(r => r.json())
        .then(data => {
          if (data.data) {
            setCensusData(data.data)
          }
        })
        .catch(err => {
          console.error('Failed to fetch census data:', err)
        })
    }
  }, [parcel])

  // Helper component to render a message bubble with optional truncation
  function MessageBubble({ msg }) {
    const [expanded, setExpanded] = useState(false)
    const limit = 350
    const isLong = typeof msg.text === 'string' && msg.text.length > limit
    const displayText = isLong && !expanded ? msg.text.slice(0, limit) + '…' : msg.text

    const commonStyle = {
      display: 'inline-block',
      maxWidth: '80%',
      wordBreak: 'break-word',
      whiteSpace: 'pre-wrap',
      fontSize: 14,
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
            <button onClick={() => setExpanded(!expanded)} style={{ background: 'transparent', border: 'none', color: isBot ? '#2563eb' : '#93c5fd', cursor: 'pointer', padding: 0 }}>
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

  if (!parcel) return null

  return (
    <div style={{
      position: 'absolute',
      right: 24,
      top: 80,
      width: 420,
      height: '600px',
      maxHeight: '75vh',
      background: 'white',
      borderRadius: 10,
      boxShadow: '0 8px 24px rgba(15, 23, 42, 0.12)',
      zIndex: 2000,
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Full feature/property summary at top */}
      <div style={{ padding: 12, borderBottom: '1px solid #eee', background: '#fbfbfb', maxHeight: '24vh', overflowY: 'auto' }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>{parcel.address || 'Selected Parcel'}</div>
        <div style={{ fontSize: 13, color: '#111', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {(() => {
            const props = parcel || {}
            const labelMap = {
              objectid: 'Object ID',
              address: 'Address',
              owner1: 'Owner',
              bldg_desc: 'Land Type',
              opa_id: 'OPA ID',
              councildistrict: 'Council District',
              zoningbasedistrict: 'Zoning',
              zipcode: 'ZIP Code',
              land_rank: 'Land Rank',
              date_update: 'Last Update',
              Shape__Area: 'Area',
              Shape__Length: 'Perimeter',
              lat: 'Latitude',
              lon: 'Longitude'
            }
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

            // Show a reduced set of summary fields to avoid cutting off the top
            const keysToShow = ['address','owner1','bldg_desc','zoningbasedistrict','councildistrict','zipcode']
            const used = new Set()
            const rows = []
            for (const k of keysToShow) {
              if (props[k] !== undefined) {
                const val = formatValue(k, props[k])
                if (val !== null) {
                  rows.push(<div key={k}><strong>{labelMap[k] || k}:</strong> {val}</div>)
                  used.add(k)
                }
              }
            }
            // extras (limit 6)
            let extraCount = 0
            const skipKeys = new Set(['lniaddresskey','build_rank'])
            for (const [k,v] of Object.entries(props)) {
              if (used.has(k)) continue
              if (extraCount >= 6) break
              if (skipKeys.has(String(k).toLowerCase())) continue
              const val = formatValue(k, v)
              if (val !== null) {
                rows.push(<div key={k}><strong>{k}:</strong> {val}</div>)
                extraCount += 1
              }
            }
            return rows
          })()}
          
          {censusData && (
            <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #eee' }}>
              <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Census Tract Data</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 13 }}>
                {censusData.category_code_description && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#64748b' }}>Property Type:</span>
                    <span style={{ fontWeight: 500 }}>{censusData.category_code_description}</span>
                  </div>
                )}
                {censusData.census_tract && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#64748b' }}>Census Tract:</span>
                    <span style={{ fontWeight: 500 }}>{censusData.census_tract}</span>
                  </div>
                )}
                {censusData.tract_total_pop !== null && censusData.tract_total_pop !== undefined && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#64748b' }}>Tract Population:</span>
                    <span style={{ fontWeight: 500 }}>{Math.round(censusData.tract_total_pop).toLocaleString()}</span>
                  </div>
                )}
                {censusData.tract_median_income !== null && censusData.tract_median_income !== undefined && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#64748b' }}>Median Income:</span>
                    <span style={{ fontWeight: 500 }}>${Math.round(censusData.tract_median_income).toLocaleString()}</span>
                  </div>
                )}
                {censusData.tract_median_age !== null && censusData.tract_median_age !== undefined && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#64748b' }}>Median Age:</span>
                    <span style={{ fontWeight: 500 }}>{censusData.tract_median_age.toFixed(1)} years</span>
                  </div>
                )}
                {censusData.tract_median_home_value !== null && censusData.tract_median_home_value !== undefined && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#64748b' }}>Median Home Value:</span>
                    <span style={{ fontWeight: 500 }}>${Math.round(censusData.tract_median_home_value).toLocaleString()}</span>
                  </div>
                )}
                {censusData.tract_median_rent !== null && censusData.tract_median_rent !== undefined && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#64748b' }}>Median Rent:</span>
                    <span style={{ fontWeight: 500 }}>${Math.round(censusData.tract_median_rent).toLocaleString()}/mo</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {scores && (
            <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #eee' }}>
              <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Geographic Scores</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 13 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#64748b' }}>Environmental:</span>
                  <span style={{ fontWeight: 500 }}>{scores.environmental_score?.toFixed(1) || 'N/A'}/10</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#64748b' }}>Recreational:</span>
                  <span style={{ fontWeight: 500 }}>{scores.recreational_score?.toFixed(1) || 'N/A'}/10</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#64748b' }}>Transit:</span>
                  <span style={{ fontWeight: 500 }}>{scores.transit_score?.toFixed(1) || 'N/A'}/10</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#64748b' }}>Walkability:</span>
                  <span style={{ fontWeight: 500 }}>{scores.walkability_score?.toFixed(1) || 'N/A'}/10</span>
                </div>
              </div>
            </div>
          )}
          
          {loadingScores && (
            <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #eee', fontSize: 13, color: '#64748b' }}>
              Loading data...
            </div>
          )}
        </div>
      </div>

      <div style={{ padding: 12, overflowY: 'auto', flex: 1 }}>
        {messages.map(msg => (
          <div key={msg.id} style={{ marginBottom: 10, display: 'flex', justifyContent: msg.sender === 'bot' ? 'flex-start' : 'flex-end' }}>
            <MessageBubble msg={msg} />
          </div>
        ))}
      </div>

      <form onSubmit={handleSend} style={{ display: 'flex', gap: 8, padding: 12, borderTop: '1px solid #eee' }}>
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask about this parcel..."
          style={{ flex: 1, padding: '8px 10px', borderRadius: 8, border: '1px solid #e2e8f0' }}
        />
        <button type="submit" className="chat-send-button">Send</button>
        <button type="button" onClick={() => onClose()} style={{ padding: '10px 12px', borderRadius: 8, border: '1px solid #e2e8f0', background: '#fff', cursor: 'pointer' }}>Close</button>
      </form>
    </div>
  )
}
