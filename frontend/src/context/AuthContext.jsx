import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { supabase } from '../lib/supabase'

const AuthContext = createContext({
  user: null,
  session: null,
  loading: true,
  login: async () => ({ error: null }),
  signup: async () => ({ error: null }),
  logout: async () => ({ error: null }),
})

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null)
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let isMounted = true

    const loadSession = async () => {
      const { data, error } = await supabase.auth.getSession()
      if (!isMounted) return
      if (!error) {
        setSession(data.session)
        setUser(data.session?.user ?? null)
      }
      setLoading(false)
    }

    loadSession()

    const { data: subscription } = supabase.auth.onAuthStateChange((_event, newSession) => {
      setSession(newSession)
      setUser(newSession?.user ?? null)
      setLoading(false)
    })

    return () => {
      isMounted = false
      subscription?.subscription?.unsubscribe()
    }
  }, [])

  const login = async (email, password) => {
    return supabase.auth.signInWithPassword({
      email,
      password,
    })
  }

  const signup = async (email, password) => {
    return supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: `${window.location.origin}/profile-setup`,
      },
    })
  }

  const logout = async () => {
    return supabase.auth.signOut()
  }

  const value = useMemo(
    () => ({
      user,
      session,
      loading,
      login,
      signup,
      logout,
    }),
    [user, session, loading]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  return useContext(AuthContext)
}
