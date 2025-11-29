import React, { useState, useRef, useEffect } from 'react'

const API_URL_CHAT = 'http://localhost:8000/chat'

const initialMessages = [
  { id: 1, text: "Hello! How can I help you with Philadelphia development questions?", sender: 'bot' }
]

const callChat = async (message, plotInfo) => {
  const response = await fetch(API_URL_CHAT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message, plotInfo }),
  })
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }
  return response.json()
}

export default function Chat({ plotInfo }) {
  const [messages, setMessages] = useState(initialMessages)
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

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
      const data = await callChat(currentInput, plotInfo)
      
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
      <div className="chat-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message ${msg.sender}`}>
            <div className="message-content">{msg.text}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input-form" onSubmit={handleSend}>
        <input
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
