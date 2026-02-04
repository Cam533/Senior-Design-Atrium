import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import Map from '../components/Map'
import Chat from '../components/Chat'
import Login from '../components/Auth/Login'
import ProfileSetup from '../components/Auth/ProfileSetup'
import atriumIcon from '../pics/atrium_icon.png'
import { useAuth } from './context/AuthContext'

function Topbar() {
  const location = useLocation()
  const { user, logout } = useAuth()
  
  return (
    <div className="topbar">
      <div className="brand">
        <img src={atriumIcon} alt="Atrium icon" className="brand-icon" />
        <div className="brand-text">atrium</div>
      </div>
      <nav className="topbar-nav">
        
        <Link 
          to="/" 
          className={location.pathname === '/' ? 'nav-link nav-button active' : 'nav-link nav-button'}
        >
          Map
        </Link>
        <Link 
          to="/chat" 
          className={location.pathname === '/chat' ? 'nav-link nav-button active' : 'nav-link nav-button'}
        >
          Chat
        </Link>
        {!user ? (
          <Link
            to="/login"
            className={location.pathname === '/login' ? 'nav-link nav-button active' : 'nav-link nav-button'}
          >
            Login
          </Link>
        ) : (
          <button className="nav-link nav-button" type="button" onClick={logout}>
            Logout
          </button>
        )}
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
          <Route path="/login" element={<Login />} />
          <Route path="/profile-setup" element={<ProfileSetup />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
