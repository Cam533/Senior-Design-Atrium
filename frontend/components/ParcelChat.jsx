import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

const API_URL = 'http://localhost:8000/chat'

// Independent ephemeral parcel chat: own messages, input, and send behavior.
// No global state is modified; closing and reopening resets the panel.
export default function ParcelChat({ parcel = null, onClose = () => {} }) {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    if (!parcel) return
    // reset messages when opening parcel chat (ephemeral)
    setMessages([])
    setInputValue('')
    setTimeout(() => inputRef.current?.focus(), 200)
  }, [parcel])

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
      <div style={{ padding: 12, borderBottom: '1px solid #eee', background: '#fbfbfb', maxHeight: '30vh', overflowY: 'auto' }}>
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
              Shape__Length: 'Perimeter'
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
              if (k === 'date_update') {
                try { const d = new Date(v); if (!isNaN(d)) return d.toLocaleDateString() } catch (e) {}
              }
              return String(v)
            }

            const keysToShow = ['address','owner1','bldg_desc','zoningbasedistrict','councildistrict','zipcode','land_rank','Shape__Area','Shape__Length','date_update','opa_id','objectid']
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
        </div>
      </div>

      <div style={{ padding: 12, overflowY: 'auto', flex: 1 }}>
        {messages.map(msg => (
          <div key={msg.id} style={{ marginBottom: 10 }}>
            {msg.sender === 'bot' ? (
              <div style={{ background: '#f1f5f9', padding: 8, borderRadius: 8 }}><ReactMarkdown>{msg.text}</ReactMarkdown></div>
            ) : (
              <div style={{ background: '#0f172a', color: 'white', padding: 8, borderRadius: 8, textAlign: 'right' }}>{msg.text}</div>
            )}
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
