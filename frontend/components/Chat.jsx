import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

const API_URL = 'http://localhost:8000/chat'

const initialMessages = [
  { id: 1, text: "Hello! How can I help you with Philadelphia development questions?", sender: 'bot' }
]
const STORAGE_KEY = 'atrium_chat_messages_v1'

const callChat = async (message) => {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  })
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }
  return response.json()
}

export default function Chat({ selectedParcel = null, onNewChat = null }) {
  const [messages, setMessages] = useState(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) return JSON.parse(raw)
    } catch (e) {
      console.warn('Failed to parse stored chat messages:', e)
    }
    return initialMessages
  })
  const [inputValue, setInputValue] = useState(() => {
    try {
      const raw = localStorage.getItem(`${STORAGE_KEY}_input`)
      if (raw) return JSON.parse(raw)
    } catch (e) {
      /* ignore */
    }
    return ''
  })
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // When selectedParcel changes, scroll chat to bottom and optionally add a small context message
  useEffect(() => {
    if (!selectedParcel) return
    // Don't inject a parcel summary message into the global chat.
    // Only focus the input for convenience when a parcel is selected.
    setTimeout(() => inputRef.current?.focus(), 200)
  }, [selectedParcel])

  // Persist messages (and input) to localStorage when they change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages))
    } catch (e) {
      console.warn('Failed to save chat messages to localStorage:', e)
    }
  }, [messages])

  useEffect(() => {
    try {
      localStorage.setItem(`${STORAGE_KEY}_input`, JSON.stringify(inputValue))
    } catch (e) {
      /* ignore */
    }
  }, [inputValue])

  const handleSend = async (e) => {
    e.preventDefault()
    if (!inputValue.trim() || isLoading) return

    // Add user message
    const userMessage = { id: Date.now(), text: inputValue, sender: 'user' }
    setMessages(prev => [...prev, userMessage])
    const currentInput = inputValue
    setInputValue('')
    setIsLoading(true)

    // Add loading message
    const loadingMessage = { 
      id: Date.now() + 1, 
      text: "Thinking...", 
      sender: 'bot',
      isLoading: true
    }
    setMessages(prev => [...prev, loadingMessage])

    try {
      // Call backend API
      const data = await callChat(currentInput)
      
      // Remove loading message and add bot response
      setMessages(prev => {
        const withoutLoading = prev.filter(msg => !msg.isLoading)
        return [...withoutLoading, {
          id: Date.now() + 2,
          text: data.message || "Sorry, I couldn't process that request.",
          sender: 'bot'
        }]
      })
    } catch (error) {
      console.error('Error calling API:', error)
      // Remove loading message and add error message
      setMessages(prev => {
        const withoutLoading = prev.filter(msg => !msg.isLoading)
        return [...withoutLoading, {
          id: Date.now() + 2,
          text: "Sorry, I'm having trouble connecting to the server. Please make sure the backend is running on http://localhost:8000",
          sender: 'bot'
        }]
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="chat-container">
      {/* Global action bar with New Chat button */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '8px 12px', borderBottom: '1px solid #eee', background: '#fbfbfb' }}>
        <button
          type="button"
          className="chat-send-button"
          onClick={() => {
            try {
              localStorage.removeItem(STORAGE_KEY)
              localStorage.removeItem(`${STORAGE_KEY}_input`)
            } catch (e) { /* ignore */ }
            setMessages(initialMessages)
            setInputValue('')
            if (typeof onNewChat === 'function') onNewChat()
            setTimeout(() => inputRef.current?.focus(), 100)
          }}
        >New Chat</button>
      </div>

      {selectedParcel && (
        <div style={{ padding: '10px', borderBottom: '1px solid #eee', background: '#fbfbfb' }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>{selectedParcel.address || 'Selected Parcel'}</div>
          <div style={{ fontSize: 13, color: '#333' }}>
            {selectedParcel.owner1 ? <div><strong>Owner:</strong> {selectedParcel.owner1}</div> : null}
            {selectedParcel.bldg_desc ? <div><strong>Land Type:</strong> {selectedParcel.bldg_desc}</div> : null}
            {selectedParcel.zoningbasedistrict ? <div><strong>Zoning:</strong> {selectedParcel.zoningbasedistrict}</div> : null}
            {selectedParcel.land_rank !== undefined ? <div><strong>Land Rank:</strong> {Number(selectedParcel.land_rank).toFixed(2)}</div> : null}
          </div>
          <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
            <button
              type="button"
              className="chat-send-button"
              onClick={() => {
                const address = selectedParcel.address || 'this parcel'
                setInputValue(`Tell me about ${address}`)
                setTimeout(() => inputRef.current?.focus(), 100)
              }}
            >Ask about this parcel</button>
          </div>
        </div>
      )}

      <div className="chat-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message ${msg.sender}`}>
            <div className="message-content">
              {msg.sender === 'bot' ? (
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              ) : (
                msg.text
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input-form" onSubmit={handleSend}>
        <input
          ref={inputRef}
          type="text"
          className="chat-input"
          placeholder="Ask about permits, zoning, or development..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
        />
        <button type="submit" className="chat-send-button">
          Send
        </button>
      </form>
    </div>
  )
}
