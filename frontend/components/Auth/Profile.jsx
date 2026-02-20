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

export default function Profile() {
  const [userType, setUserType] = useState('')
  const [organization, setOrganization] = useState('')
  const [neighborhood, setNeighborhood] = useState('')
  const [otherSpecify, setOtherSpecify] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingProfile, setLoadingProfile] = useState(true)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (!user) {
      navigate('/login')
      return
    }

    // Load existing profile data
    const loadProfile = async () => {
      try {
        const { data, error } = await supabase
          .from('users')
          .select('*')
          .eq('id', user.id)
          .single()

        if (error && error.code !== 'PGRST116') {
          throw error
        }

        if (data) {
          setUserType(data.user_type || '')
          setOrganization(data.organization || '')
          setNeighborhood(data.neighborhood || '')
          setOtherSpecify(data.other_specify || '')
        }
      } catch (err) {
        console.error('Error loading profile:', err)
      } finally {
        setLoadingProfile(false)
      }
    }

    loadProfile()
  }, [user, navigate])

  const handleUpdateProfile = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setMessage('')

    try {
      const { error: upsertError } = await supabase
        .from('users')
        .upsert({
          id: user.id,
          email: user.email,
          user_type: userType,
          organization: organization || null,
          neighborhood: neighborhood || null,
          other_specify: userType === 'other' ? otherSpecify : null,
          profile_complete: true,
        })

      if (upsertError) throw upsertError
      setMessage('Profile updated successfully!')
    } catch (err) {
      setError(err.message)
    }

    setLoading(false)
  }

  const handleChangePassword = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setMessage('')

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match')
      setLoading(false)
      return
    }

    if (newPassword.length < 6) {
      setError('Password must be at least 6 characters')
      setLoading(false)
      return
    }

    try {
      const { error: updateError } = await supabase.auth.updateUser({
        password: newPassword,
      })

      if (updateError) throw updateError

      setMessage('Password changed successfully!')
      setNewPassword('')
      setConfirmPassword('')
    } catch (err) {
      setError(err.message)
    }

    setLoading(false)
  }

  const handleLogout = async () => {
    await logout()
    navigate('/login')
  }

  if (loadingProfile) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <p>Loading profile...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h1 className="brand-auth">Your Profile</h1>
        <p>Manage your account settings and preferences.</p>

        {/* Profile Information Section */}
        <form onSubmit={handleUpdateProfile} className="auth-form">
          <h3>Profile Information</h3>
          
          <div className="form-group">
            <label>Email</label>
            <input
              type="email"
              value={user?.email || ''}
              disabled
              style={{ backgroundColor: '#f1f5f9', cursor: 'not-allowed' }}
            />
          </div>

          <div className="form-group">
            <label htmlFor="userType">User Type</label>
            <div className="user-type-options">
              {USER_TYPES.map((type) => (
                <label key={type.value} className="radio-option">
                  <input
                    type="radio"
                    name="userType"
                    value={type.value}
                    checked={userType === type.value}
                    onChange={(e) => setUserType(e.target.value)}
                  />
                  <span>{type.label}</span>
                </label>
              ))}
            </div>
          </div>

          {(userType === 'business' || userType === 'nonprofit') && (
            <div className="form-group">
              <label>Organization</label>
              <input
                type="text"
                value={organization}
                onChange={(e) => setOrganization(e.target.value)}
                placeholder="Organization name"
              />
            </div>
          )}

          {userType === 'other' && (
            <div className="form-group">
              <label>Please Specify</label>
              <input
                type="text"
                value={otherSpecify}
                onChange={(e) => setOtherSpecify(e.target.value)}
                placeholder="Specify your user type"
              />
            </div>
          )}

          <div className="form-group">
            <label>Neighborhood / Area of Interest</label>
            <input
              type="text"
              value={neighborhood}
              onChange={(e) => setNeighborhood(e.target.value)}
              placeholder="Your neighborhood or area"
            />
          </div>

          <button type="submit" disabled={loading}>
            {loading ? 'Updating...' : 'Update Profile'}
          </button>
        </form>

        {/* Change Password Section */}
        <form onSubmit={handleChangePassword} className="auth-form" style={{ marginTop: '24px' }}>
          <h3 style={{ marginBottom: '8px' }}>Change Password</h3>
          
          <input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            placeholder="New password"
            disabled={loading}
            style={{ marginBottom: '8px' }}
          />

          <input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="Confirm new password"
            disabled={loading}
            style={{ marginBottom: '12px' }}
          />

          <button type="submit" disabled={loading}>
            {loading ? 'Changing...' : 'Change Password'}
          </button>
        </form>

        {message && <p className="auth-message" style={{ color: '#059669' }}>{message}</p>}
        {error && <p className="auth-error">{error}</p>}

        {/* Logout Button */}
        <button 
          onClick={handleLogout}
          style={{ 
            marginTop: '24px', 
            width: '100%',
            padding: '12px 14px',
            border: '1px solid #dc2626',
            borderRadius: '8px',
            background: '#fff',
            color: '#dc2626',
            fontWeight: '600',
            cursor: 'pointer',
            fontSize: '0.95rem',
            transition: 'all 0.2s'
          }}
          onMouseOver={(e) => {
            e.target.style.background = '#dc2626'
            e.target.style.color = '#fff'
          }}
          onMouseOut={(e) => {
            e.target.style.background = '#fff'
            e.target.style.color = '#dc2626'
          }}
        >
          Logout
        </button>
      </div>
    </div>
  )
}
