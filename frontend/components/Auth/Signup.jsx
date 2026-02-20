import { useEffect, useState } from 'react'
import { useAuth } from '../../src/context/AuthContext'
import { useNavigate, Link } from 'react-router-dom'
import '../../src/styles/Auth.css'

const AVATAR_BUCKET = 'avatars'
const MAX_AVATAR_SIZE_BYTES = 2 * 1024 * 1024 // 2MB
const ALLOWED_AVATAR_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']

function getInitials(firstName, lastName, email) {
  const placeholderLike = (s) => /^(first|last)(\s+name)?$/i.test((s || '').trim())
  const first = (firstName || '').trim()
  const last = (lastName || '').trim()
  const hasFirst = first && !placeholderLike(first)
  const hasLast = last && !placeholderLike(last)
  if (hasFirst && hasLast) return `${first[0]}${last[0]}`.toUpperCase()
  if (hasFirst) return first[0].toUpperCase()
  if (email) return email[0].toUpperCase()
  return '?'
}

export default function Signup() {
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [avatarFile, setAvatarFile] = useState(null)
  const [avatarPreviewUrl, setAvatarPreviewUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const { signup, user } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (user) {
      navigate('/profile-setup', {
        state: {
          firstName: firstName.trim() || undefined,
          lastName: lastName.trim() || undefined,
          avatarFile: avatarFile || undefined,
        },
      })
    }
  }, [user, navigate])

  const handleAvatarChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!ALLOWED_AVATAR_TYPES.includes(file.type)) {
      setMessage('Please choose a JPEG, PNG, GIF, or WebP image.')
      return
    }
    if (file.size > MAX_AVATAR_SIZE_BYTES) {
      setMessage('Image must be 2MB or smaller.')
      return
    }
    setMessage('')
    if (avatarPreviewUrl && avatarPreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(avatarPreviewUrl)
    }
    setAvatarFile(file)
    setAvatarPreviewUrl(URL.createObjectURL(file))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setMessage('')

    const { error } = await signup(email, password)

    if (error) {
      setMessage(`Error: ${error.message}`)
    } else {
      setMessage('Account created! Redirecting...')
    }

    setLoading(false)
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h1 className="brand-auth">Create your account</h1>

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="profile-avatar-section">
            <div className="profile-avatar-wrap">
              {avatarPreviewUrl ? (
                <img src={avatarPreviewUrl} alt="Profile" className="profile-avatar profile-avatar-img" />
              ) : (
                <div className="profile-avatar profile-avatar-initials" aria-hidden>
                  {getInitials(firstName, lastName, email)}
                </div>
              )}
            </div>
            <div className="profile-avatar-actions">
              <label className="profile-avatar-upload-label">
                <input
                  type="file"
                  accept={ALLOWED_AVATAR_TYPES.join(',')}
                  onChange={handleAvatarChange}
                  disabled={loading}
                  className="profile-avatar-input"
                />
                {avatarFile ? 'Change photo' : 'Profile pic'}
              </label>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="signup-firstName">First name</label>
              <input
                id="signup-firstName"
                type="text"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                placeholder="First name"
                required
                disabled={loading}
              />
            </div>
            <div className="form-group">
              <label htmlFor="signup-lastName">Last name</label>
              <input
                id="signup-lastName"
                type="text"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                placeholder="Last name"
                disabled={loading}
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="signup-email">Email</label>
            <input
              id="signup-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email"
              required
              disabled={loading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="signup-password">Password</label>
            <input
              id="signup-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Create a password"
              required
              disabled={loading}
            />
          </div>

          <button type="submit" disabled={loading}>
            {loading ? 'Creating account...' : 'Sign Up'}
          </button>
        </form>

        {message && <p className="auth-message">{message}</p>}

        <p className="auth-footer">
          Already have an account? <Link to="/login">Log in</Link>
        </p>
      </div>
    </div>
  )
}
