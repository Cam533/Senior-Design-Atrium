import { useEffect, useState } from 'react'
import { useAuth } from '../../src/context/AuthContext'
import { useNavigate } from 'react-router-dom'
import '../../src/styles/Auth.css'

export default function Login() {
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const { login, user } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (user) {
      navigate('/')
    }
  }, [user, navigate])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setMessage('')

    const { error } = await login(email)

    if (error) {
      setMessage(`Error: ${error.message}`)
    } else {
      setMessage('Check your email for a login link!')
      setEmail('')
    }

    setLoading(false)
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h1 className='brand-auth'>Login to Atrium</h1>
        <p>Sign in to save parcels, create projects, and build your portfolio.</p>
        
        <form onSubmit={handleSubmit} className="auth-form">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            required
            disabled={loading}
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Sending...' : 'Send Login Link'}
          </button>
        </form>

        {message && <p className="auth-message">{message}</p>}

        <p className="auth-footer">
          Or continue as a guest to explore the map and chat.
        </p>
        <button 
          className="auth-guest-btn"
          onClick={() => navigate('/')}
        >
          Continue as Guest
        </button>
      </div>
    </div>
  )
}
