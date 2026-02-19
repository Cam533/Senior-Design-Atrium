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
  const [deleting, setDeleting] = useState(false)
  const [activeTab, setActiveTab] = useState('profile')
  const { user, session, logout } = useAuth()
  const navigate = useNavigate()

  const TABS = [
    { id: 'profile', label: 'Profile' },
    { id: 'preferences', label: 'Preferences' },
    { id: 'account', label: 'Security' },
  ]

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

  const handleDeleteAccount = async () => {
    const token = session?.access_token
    if (!token) {
      setError('Not signed in.')
      return
    }
    setDeleting(true)
    setError('')
    try {
      const res = await fetch('http://localhost:8000/delete-account', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(data.detail || res.statusText || 'Failed to delete account')
      }
      await logout()
      navigate('/')
    } catch (err) {
      setError(err.message)
    } finally {
      setDeleting(false)
    }
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
      <div className="profile-page">
        <nav className="profile-tabs">
          <h1 className="profile-tabs-title">Your Profile</h1>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`profile-tab ${activeTab === tab.id ? 'profile-tab-active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
        <div className="profile-content">
          {activeTab === 'profile' && (
            <>
              <p className="profile-intro">Manage your account settings and preferences.</p>
              {user?.created_at && (
                <p className="profile-date">Account created: {new Date(user.created_at).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</p>
              )}
              {user?.last_sign_in_at && (
                <p className="profile-date">Last signed in: {new Date(user.last_sign_in_at).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</p>
              )}

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
            </>
          )}

          {activeTab === 'preferences' && (
            <>
              <h2 className="profile-content-heading">Preferences</h2>
              <p className="profile-date">
                Notification and display preferences can be configured here.
              </p>
            </>
          )}

          {activeTab === 'account' && (
            <>
              <h2 className="profile-content-heading">Security</h2>
              <p className="profile-delete-warning" style={{ marginBottom: '20px' }}>
                Change your password, sign out, or permanently delete your account.
              </p>

              <h3 className="profile-section-label">Change password</h3>
              <form onSubmit={handleChangePassword} className="auth-form profile-security-form">
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="New password"
                  disabled={loading}
                />
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirm new password"
                  disabled={loading}
                />
                <button type="submit" disabled={loading}>
                  {loading ? 'Changing...' : 'Change Password'}
                </button>
              </form>

              <h3 className="profile-section-label">Logout</h3>
              <button type="button" onClick={handleLogout} className="auth-logout-btn">
                Logout
              </button>

              <h3 className="profile-section-label">Delete account</h3>
              <p className="profile-delete-warning">
                This permanently deletes your account and cannot be undone.
              </p>
              <button
                type="button"
                onClick={handleDeleteAccount}
                disabled={deleting}
                className="auth-logout-btn"
                style={{ marginTop: '8px' }}
              >
                {deleting ? 'Deleting...' : 'Permanently delete my account'}
              </button>
            </>
          )}

          {message && <p className="auth-message" style={{ color: '#059669' }}>{message}</p>}
          {error && <p className="auth-error">{error}</p>}
        </div>
      </div>
    </div>
  )
}
