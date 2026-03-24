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

const AVATAR_BUCKET = 'avatars'
const MAX_AVATAR_SIZE_BYTES = 2 * 1024 * 1024 // 2MB
const ALLOWED_AVATAR_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']

function getInitials(firstName, lastName) {
  const placeholderLike = (s) => /^(first|last)(\s+name)?$/i.test((s || '').trim())
  const first = (firstName || '').trim()
  const last = (lastName || '').trim()
  const hasFirst = first && !placeholderLike(first)
  const hasLast = last && !placeholderLike(last)
  if (hasFirst && hasLast) return `${first[0]}${last[0]}`.toUpperCase()
  if (hasFirst) return first[0].toUpperCase()
  return ''
}

export default function Profile() {
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [avatarUrl, setAvatarUrl] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [userType, setUserType] = useState('')
  const [organization, setOrganization] = useState('')
  const [neighborhood, setNeighborhood] = useState('')
  const [otherSpecify, setOtherSpecify] = useState('')
  const [notifyPlotLikes, setNotifyPlotLikes] = useState(true)
  const [notifyPlotImages, setNotifyPlotImages] = useState(false)
  const [muteAll, setMuteAll] = useState(false)
  const [loading, setLoading] = useState(false)
  const [savingPreferences, setSavingPreferences] = useState(false)
  const [loadingProfile, setLoadingProfile] = useState(true)
  const [uploadingAvatar, setUploadingAvatar] = useState(false)
  const [message, setMessage] = useState('')
  const [messageTab, setMessageTab] = useState(null) // which tab the message belongs to
  const [error, setError] = useState('')
  const [deleting, setDeleting] = useState(false)
  const [activeTab, setActiveTab] = useState('profile')
  const { user, session, logout } = useAuth()
  const navigate = useNavigate()
  const initials = getInitials(firstName, lastName)

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
          setFirstName(data.first_name || '')
          setLastName(data.last_name || '')
          setAvatarUrl(data.avatar_url || '')
          setUserType(data.user_type || '')
          setOrganization(data.organization || '')
          setNeighborhood(data.neighborhood || '')
          setOtherSpecify(data.other_specify || '')
          setNotifyPlotLikes(data.email_plot_updates !== false)
          setNotifyPlotImages(data.email_product_news === true)
          setMuteAll(data.unsubscribe_all === true)
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
    const trimmedFirstName = firstName.trim()
    const trimmedLastName = lastName.trim()

    if (!trimmedFirstName || !trimmedLastName) {
      setError('First name and last name are required.')
      setLoading(false)
      return
    }

    try {
      const { error: upsertError } = await supabase
        .from('users')
        .upsert({
          id: user.id,
          email: user.email,
          first_name: trimmedFirstName,
          last_name: trimmedLastName,
          avatar_url: avatarUrl || null,
          user_type: userType,
          organization: organization || null,
          neighborhood: neighborhood || null,
          other_specify: userType === 'other' ? otherSpecify : null,
          profile_complete: true,
        })

      if (upsertError) throw upsertError
      setMessage('Profile updated successfully!')
      setMessageTab('profile')
    } catch (err) {
      setError(err.message)
    }

    setLoading(false)
  }

  const handleAvatarUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file || !user?.id) return

    if (!ALLOWED_AVATAR_TYPES.includes(file.type)) {
      setError('Please choose a JPEG, PNG, GIF, or WebP image.')
      return
    }
    if (file.size > MAX_AVATAR_SIZE_BYTES) {
      setError('Image must be 2MB or smaller.')
      return
    }

    setUploadingAvatar(true)
    setError('')
    setMessage('')

    try {
      const ext = file.name.split('.').pop()?.toLowerCase() || 'jpg'
      const path = `${user.id}.${ext}`

      const { error: uploadError } = await supabase.storage
        .from(AVATAR_BUCKET)
        .upload(path, file, { upsert: true, contentType: file.type })

      if (uploadError) throw uploadError

      const { data: urlData } = supabase.storage.from(AVATAR_BUCKET).getPublicUrl(path)
      const newUrl = urlData?.publicUrl ?? ''

      const { data: updatedRow, error: updateError } = await supabase
        .from('users')
        .update({ avatar_url: newUrl })
        .eq('id', user.id)
        .select('avatar_url')
        .single()

      if (updateError) throw updateError
      if (!updatedRow?.avatar_url) {
        throw new Error('Profile could not be updated. Check that you have permission to update your profile.')
      }

      setAvatarUrl(updatedRow.avatar_url)
      setMessage('Profile picture updated.')
      setMessageTab('profile')
    } catch (err) {
      setError(err.message)
    } finally {
      setUploadingAvatar(false)
      e.target.value = ''
    }
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
      setMessageTab('account')
      setNewPassword('')
      setConfirmPassword('')
    } catch (err) {
      setError(err.message)
    }

    setLoading(false)
  }

  const handleUpdatePreferences = async (e) => {
    e.preventDefault()
    setSavingPreferences(true)
    setError('')
    setMessage('')
    try {
      const { error: updateError } = await supabase
        .from('users')
        .update({
          email_plot_updates: muteAll ? false : notifyPlotLikes,
          email_product_news: muteAll ? false : notifyPlotImages,
          unsubscribe_all: muteAll,
        })
        .eq('id', user.id)

      if (updateError) throw updateError
      setMessage('Preferences saved.')
      setMessageTab('preferences')
    } catch (err) {
      setError(err.message)
    }
    setSavingPreferences(false)
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

          <div className="profile-avatar-section">
            <div className="profile-avatar-wrap">
              {avatarUrl ? (
                <img src={avatarUrl} alt="Profile" className="profile-avatar profile-avatar-img" />
              ) : initials ? (
                <div className="profile-avatar profile-avatar-initials" aria-hidden>
                  {initials}
                </div>
              ) : (
                <div className="profile-avatar" aria-hidden />
              )}
            </div>
            <div className="profile-avatar-actions">
              <label className="profile-avatar-upload-label">
                <input
                  type="file"
                  accept={ALLOWED_AVATAR_TYPES.join(',')}
                  onChange={handleAvatarUpload}
                  disabled={uploadingAvatar}
                  className="profile-avatar-input"
                />
                {uploadingAvatar ? 'Uploading...' : 'Upload photo'}
              </label>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="firstName">First Name</label>
              <input
                id="firstName"
                type="text"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                placeholder="First Name"
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="lastName">Last Name</label>
              <input
                id="lastName"
                type="text"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                placeholder="Last Name"
                required
              />
            </div>
          </div>
          
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

          <button type="submit" disabled={loading || !firstName.trim() || !lastName.trim()}>
            {loading ? 'Updating...' : 'Update Profile'}
          </button>
        </form>
            </>
          )}

          {activeTab === 'preferences' && (
            <>
              <h2 className="profile-content-heading">Notifications</h2>
              <p className="profile-date" style={{ marginBottom: '20px' }}>
                Choose which in-website notifications you receive.
              </p>
              <form onSubmit={handleUpdatePreferences} className="auth-form">
                <div className="form-group">
                  <label className="checkbox-option">
                    <input
                      type="checkbox"
                      checked={!muteAll && notifyPlotLikes}
                      onChange={(e) => setNotifyPlotLikes(e.target.checked)}
                      disabled={muteAll}
                    />
                    <span>Notify me when someone likes a lot I saved</span>
                  </label>
                  <p className="profile-pref-hint">Get notified when another user likes a parcel you have saved.</p>
                </div>
                <div className="form-group">
                  <label className="checkbox-option">
                    <input
                      type="checkbox"
                      checked={!muteAll && notifyPlotImages}
                      onChange={(e) => setNotifyPlotImages(e.target.checked)}
                      disabled={muteAll}
                    />
                    <span>Notify me when a new photo is added to a lot I saved</span>
                  </label>
                  <p className="profile-pref-hint">Get notified when someone uploads a photo to a parcel you have saved.</p>
                </div>
                <div className="form-group">
                  <label className="checkbox-option">
                    <input
                      type="checkbox"
                      checked={muteAll}
                      onChange={(e) => setMuteAll(e.target.checked)}
                    />
                    <span>Mute all notifications</span>
                  </label>
                  <p className="profile-pref-hint">Turn off all in-website notifications.</p>
                </div>
                <button type="submit" disabled={savingPreferences}>
                  {savingPreferences ? 'Saving...' : 'Save preferences'}
                </button>
              </form>
            </>
          )}

          {activeTab === 'account' && (
            <>
              <h2 className="profile-content-heading">Security</h2>
              <p className="profile-date" style={{ marginBottom: '20px' }}>
                Manage your password, sessions, and account deletion.
              </p>

              <h3 className="profile-section-label">Change Password</h3>
              <form onSubmit={handleChangePassword} className="auth-form">
                <div className="form-group">
                  <label htmlFor="newPassword">New Password</label>
                  <input
                    id="newPassword"
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Enter new password"
                    disabled={loading}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="confirmPassword">Confirm Password</label>
                  <input
                    id="confirmPassword"
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm new password"
                    disabled={loading}
                  />
                </div>
                <button type="submit" disabled={loading}>
                  {loading ? 'Changing...' : 'Change Password'}
                </button>
              </form>

              <h3 className="profile-section-label" style={{ marginTop: '30px' }}>Logout</h3>
              <p className="profile-date" style={{ marginBottom: '12px' }}>Sign out from this device.</p>
              <button type="button" onClick={handleLogout} className="auth-logout-btn">
                Logout
              </button>

              <h3 className="profile-section-label" style={{ marginTop: '30px' }}>Delete Account</h3>
              <p className="profile-delete-warning">
                Permanently delete your account and all associated data. This action cannot be undone.
              </p>
              <button
                type="button"
                onClick={handleDeleteAccount}
                disabled={deleting}
                className="auth-logout-btn"
                style={{ marginTop: '12px', backgroundColor: '#dc2626', color: '#fff' }}
              >
                {deleting ? 'Deleting...' : 'Permanently Delete Account'}
              </button>
            </>
          )}

          {message && messageTab === activeTab && <p className="auth-message" style={{ color: '#059669' }}>{message}</p>}
          {error && <p className="auth-error">{error}</p>}
        </div>
      </div>
    </div>
  )
}
