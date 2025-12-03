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
  const [showRecommendations, setShowRecommendations] = useState(true)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Hide recommendations once the user has asked any question
  useEffect(() => {
    const hasUserMessage = messages.some(m => m.sender === 'user')
    setShowRecommendations(!hasUserMessage)
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
    setShowRecommendations(false)
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

  const startRecommendedChat = async (question) => {
    if (!question || isLoading) return
    // hide the recommendation bubble immediately
    setShowRecommendations(false)
    // Add user message
    const userMessage = { id: Date.now(), text: question, sender: 'user' }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    // Add loading message
    const loadingMessage = {
      id: Date.now() + 1,
      text: 'Thinking...',
      sender: 'bot',
      isLoading: true,
    }
    setMessages(prev => [...prev, loadingMessage])

    try {
      const data = await callChat(question)
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
      setTimeout(() => inputRef.current?.focus(), 100)
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
              setShowRecommendations(true)
            if (typeof onNewChat === 'function') onNewChat()
            setTimeout(() => inputRef.current?.focus(), 100)
          }}
        >+ New Chat</button>
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

      {/* recommendations will be rendered below the first message inside the messages area */}

      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <React.Fragment key={msg.id}>
            <div className={`chat-message ${msg.sender}`}>
              <div className="message-content">
                {msg.sender === 'bot' ? (
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                ) : (
                  msg.text
                )}
              </div>
            </div>

            {idx === 0 && !selectedParcel && showRecommendations && (
               <div style={{ display: 'flex', justifyContent: 'flex-end', padding: 12 }}>
                <div style={{ background: '#f6f9fb', borderRadius: 12, padding: 10, maxWidth: '100%' }} className="recommendation-inner">
                  <div style={{ fontSize: 12, color: '#52606d', marginBottom: 8, textAlign: 'right' }}>Recommended topics to ask about:</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                    <button
                      type="button"
                      className="chat-send-button recommendation-button"
                      onClick={() => startRecommendedChat('What are current development incentives in Philadelphia?')}
                      style={{ padding: '6px 10px', borderRadius: 999, border: '1px solid #e6edf3', background: 'white', cursor: 'pointer', fontSize: 13, color: '#0f172a' }}
                    >Development incentives</button>

                    <button
                      type="button"
                      className="chat-send-button recommendation-button"
                      onClick={() => startRecommendedChat('Where are neighborhoods seeing the most residential development recently?')}
                      style={{ padding: '6px 10px', borderRadius: 999, border: '1px solid #e6edf3', background: 'white', cursor: 'pointer', fontSize: 13, color: '#0f172a' }}
                    >Neighborhood trends</button>

                    <button
                      type="button"
                      className="chat-send-button recommendation-button"
                      onClick={() => startRecommendedChat('What are common zoning constraints for new developments in the city?')}
                      style={{ padding: '6px 10px', borderRadius: 999, border: '1px solid #e6edf3', background: 'white', cursor: 'pointer', fontSize: 13, color: '#0f172a' }}
                    >Zoning constraints</button>
                  </div>
                </div>
              </div>
            )}
          </React.Fragment>
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
