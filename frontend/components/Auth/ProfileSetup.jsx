import { useState, useEffect } from 'react'
import { useAuth } from '../../src/context/AuthContext'
import { useNavigate, useLocation, Link } from 'react-router-dom'
import { supabase } from '../../src/lib/supabase'
import '../../src/styles/Auth.css'

const USER_TYPES = [
  { value: 'resident', label: 'Personal User / Resident' },
  { value: 'business', label: 'Private Business-Affiliated User' },
  { value: 'city_planner', label: 'City Planner / Government' },
  { value: 'nonprofit', label: 'Non-profit Organization' },
  { value: 'other', label: 'Other' },
]

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

export default function ProfileSetup() {
  const location = useLocation()
  const stateFromSignup = location.state && typeof location.state === 'object' ? location.state : {}
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [firstName, setFirstName] = useState(stateFromSignup.firstName ?? '')
  const [lastName, setLastName] = useState(stateFromSignup.lastName ?? '')
  const [avatarFile, setAvatarFile] = useState(stateFromSignup.avatarFile ?? null)
  const [avatarPreviewUrl, setAvatarPreviewUrl] = useState(
    stateFromSignup.avatarFile ? URL.createObjectURL(stateFromSignup.avatarFile) : ''
  )
  const [userType, setUserType] = useState('')
  const [organization, setOrganization] = useState('')
  const [neighborhood, setNeighborhood] = useState('')
  const [otherSpecify, setOtherSpecify] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const { user, signup } = useAuth()
  const navigate = useNavigate()
  const isNewSignup = !user

  const handleAvatarChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!ALLOWED_AVATAR_TYPES.includes(file.type)) {
      setError('Please choose a JPEG, PNG, GIF, or WebP image.')
      return
    }
    if (file.size > MAX_AVATAR_SIZE_BYTES) {
      setError('Image must be 2MB or smaller.')
      return
    }
    setError('')
    if (avatarPreviewUrl && avatarPreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(avatarPreviewUrl)
    }
    setAvatarFile(file)
    setAvatarPreviewUrl(URL.createObjectURL(file))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      let targetUser = user
      if (isNewSignup) {
        const { data, error: signUpError } = await signup(email, password)
        if (signUpError) throw signUpError
        targetUser = data?.user ?? null
        if (!targetUser) {
          setError('Account created. Please check your email to confirm, then log in.')
          setLoading(false)
          return
        }
        // So the next Supabase calls (storage, users upsert) use the new session and pass RLS
        if (data?.session) {
          await supabase.auth.setSession({
            access_token: data.session.access_token,
            refresh_token: data.session.refresh_token,
          })
        }
      }

      let avatarUrl = null
      if (avatarFile && targetUser?.id) {
        const ext = avatarFile.name.split('.').pop()?.toLowerCase() || 'jpg'
        const path = `${targetUser.id}.${ext}`
        const { error: uploadError } = await supabase.storage
          .from(AVATAR_BUCKET)
          .upload(path, avatarFile, { upsert: true, contentType: avatarFile.type })
        if (uploadError) throw uploadError
        const { data: urlData } = supabase.storage.from(AVATAR_BUCKET).getPublicUrl(path)
        avatarUrl = urlData?.publicUrl ?? ''
      }

      const timestamp = new Date().toISOString()
      const { error: supabaseError } = await supabase
        .from('users')
        .upsert({
          id: targetUser.id,
          email: targetUser.email,
          first_name: firstName.trim() || null,
          last_name: lastName.trim() || null,
          avatar_url: avatarUrl,
          user_type: userType,
          organization: organization || null,
          neighborhood: neighborhood || null,
          other_specify: userType === 'other' ? otherSpecify : null,
          profile_complete: true,
          created_at: timestamp,
        })

      if (supabaseError) throw supabaseError

      navigate('/')
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Request timed out.')
      } else {
        setError(err.message)
      }
    }

    setLoading(false)
  }

  const handleSkip = () => {
    navigate('/')
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h1 className="brand-auth">Complete Your Profile</h1>

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="profile-avatar-section">
            <div className="profile-avatar-wrap">
              {avatarPreviewUrl ? (
                <img src={avatarPreviewUrl} alt="Profile" className="profile-avatar profile-avatar-img" />
              ) : (
                <div className="profile-avatar profile-avatar-initials" aria-hidden>
                  {getInitials(firstName, lastName, isNewSignup ? email : user?.email)}
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
                {avatarFile ? 'Change photo' : 'Upload photo'}
              </label>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="firstName">First name (optional)</label>
              <input
                id="firstName"
                type="text"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                placeholder="First name"
              />
            </div>
            <div className="form-group">
              <label htmlFor="lastName">Last name (optional)</label>
              <input
                id="lastName"
                type="text"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                placeholder="Last name"
              />
            </div>
          </div>

          {isNewSignup && (
            <>
              <div className="form-group">
                <label htmlFor="profile-setup-email">Email</label>
                <input
                  id="profile-setup-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Email"
                  required
                  disabled={loading}
                />
              </div>
              <div className="form-group">
                <label htmlFor="profile-setup-password">Password</label>
                <input
                  id="profile-setup-password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Create a password"
                  required
                  disabled={loading}
                />
              </div>
            </>
          )}

          <div className="form-group">
            <label htmlFor="userType">How do you describe yourself?</label>
            <div className="user-type-options">
              {USER_TYPES.map((type) => (
                <label key={type.value} className="radio-option">
                  <input
                    type="radio"
                    name="userType"
                    value={type.value}
                    checked={userType === type.value}
                    onChange={(e) => setUserType(e.target.value)}
                    required
                  />
                  <span>{type.label}</span>
                </label>
              ))}
            </div>
          </div>

          {(userType === 'business' || userType === 'nonprofit') && (
            <div className="form-group-input">
              <input
                type="text"
                value={organization}
                onChange={(e) => setOrganization(e.target.value)}
                placeholder="Organization name (optional)"
                style={{ width: '50%' }}
              />
            </div>
          )}

          {userType === 'other' && (
            <div className="form-group-input">
              <input
                type="text"
                value={otherSpecify}
                onChange={(e) => setOtherSpecify(e.target.value)}
                style={{ width: '50%' }}
                placeholder="Please specify"
                required
              />
            </div>
          )}

          <div className="auth-form">
            <label htmlFor="userType">Enter your neighborhood or area of interest:</label>
            <input
              type="text"
              value={neighborhood}
              onChange={(e) => setNeighborhood(e.target.value)}
              placeholder="Neighborhood / Area of interest (optional)"
              style={{ width: '75%' }}
            />
          </div>

          {error && <p className="auth-error">{error}</p>}

          <button type="submit" disabled={loading || !userType || (isNewSignup && (!email.trim() || !password))}>
            {loading ? 'Saving...' : 'Complete Profile'}
          </button>
        </form>

        {isNewSignup && (
          <p className="auth-footer">
            Already have an account? <Link to="/login">Log in</Link>
          </p>
        )}
        {user && (
          <button className="auth-skip-btn" onClick={handleSkip}>
            Skip for Now
          </button>
        )}
      </div>
    </div>
  )
}
