import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Map from '../components/Map'
import Chat from '../components/Chat'

function Topbar() {
  const location = useLocation()
  
  return (
    <div className="topbar">
      <div className="brand">Atrium</div>
      <nav className="topbar-nav">
        <Link 
          to="/" 
          className={location.pathname === '/' ? 'nav-link active' : 'nav-link'}
        >
          Map
        </Link>
        <Link 
          to="/chat" 
          className={location.pathname === '/chat' ? 'nav-link active' : 'nav-link'}
        >
          Chat
        </Link>
      </nav>
    </div>
  )
}

function MapPage() {
  return (
    <main className="main-content">
      <div className="map-container">
        <Map />
      </div>
    </main>
  )
}

function ChatPage() {
  return (
    <main className="main-content">
      <div className="chat-container">
        <Chat />
      </div>
    </main>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-root">
        <Topbar />
        <Routes>
          <Route path="/" element={<MapPage />} />
          <Route path="/chat" element={<ChatPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
