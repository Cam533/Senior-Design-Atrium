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
      const timestamp = new Date().toISOString()
      const { error: supabaseError } = await supabase
        .from('users')
        .upsert({
          id: user.id,
          email: user.email,
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
      setError(err.message)
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

        <button type="button" className="auth-skip-btn" onClick={handleSkip} disabled={loading}>
          {loading ? 'Saving...' : 'Skip for Now'}
        </button>
      </div>
    </div>
  )
}
