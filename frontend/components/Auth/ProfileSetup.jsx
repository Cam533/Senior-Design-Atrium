import { useState, useEffect } from 'react'
import { useAuth } from '../../src/context/AuthContext'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../../src/lib/supabase'
import '../../src/styles/Auth.css'

const USER_TYPES = [
  { value: 'resident', label: 'Personal User / Resident' },
  { value: 'business', label: 'Private Business-Affiliated User' },
  { value: 'city_planner', label: 'City Planner / Government' },
  { value: 'nonprofit', label: 'Non-profit Organization' },
  { value: 'other', label: 'Other' },
]

export default function ProfileSetup() {
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [userType, setUserType] = useState('')
  const [organization, setOrganization] = useState('')
  const [neighborhood, setNeighborhood] = useState('')
  const [otherSpecify, setOtherSpecify] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const { user } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    // Redirect if no user
    if (!user) {
      navigate('/login')
    }
  }, [user, navigate])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      // Create user profile in database
      console.log("Creating user profile in database")
      const timestamp = new Date().toISOString()
      const { error: supabaseError } = await supabase
        .from('users')
        .upsert({
          id: user.id,
          email: user.email,
          first_name: firstName.trim() || null,
          last_name: lastName.trim() || null,
          user_type: userType,
          organization: organization || null,
          neighborhood: neighborhood || null,
          other_specify: userType === 'other' ? otherSpecify : null,
          profile_complete: true,
          created_at: timestamp, // use time stamp instead of date
        })      

      if (supabaseError) throw supabaseError

      // Timeout so we don't hang if backend or DB is unreachable
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 15000)
      const res = await fetch('http://localhost:8000/add-aws-user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: user.id,
          email: user.email,
          user_type: userType,
          organization: organization || null,
          neighborhood: neighborhood || null,
          other_specify: userType === 'other' ? otherSpecify : null,
          created_at: timestamp,
        }),
        signal: controller.signal,
      })
      clearTimeout(timeoutId)

      console.log("added to aws response:", res)

      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || 'Failed to save profile')
      }


      navigate('/')
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Is the backend running? Is the database reachable (e.g. VPN)?')
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
        <p>Help us understand your needs (you can skip this for now).</p>

        <form onSubmit={handleSubmit} className="auth-form">
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

          <button type="submit" disabled={loading || !userType}>
            {loading ? 'Saving...' : 'Complete Profile'}
          </button>
        </form>

        <button className="auth-skip-btn" onClick={handleSkip}>
          Skip for Now
        </button>
      </div>
    </div>
  )
}
