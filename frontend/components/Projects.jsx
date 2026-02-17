import { useState, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { useAuth } from '../src/context/AuthContext'

export default function Projects() {
  const { user, loading } = useAuth()

  const [projects, setProjects] = useState([]) // start empty
  const [showForm, setShowForm] = useState(false)
  const [projectName, setProjectName] = useState('')
  const [projectDescription, setProjectDescription] = useState('')

  // ✅ hooks must be before any return
  useEffect(() => {
    if (!user) return

    const fetchProjects = async () => {
      try {
        const response = await fetch(
          `http://localhost:8000/projects?owner_id=${user.id}`
        )

        if (!response.ok) {
          console.error('Failed to fetch projects')
          return
        }

        const data = await response.json()
        setProjects(Array.isArray(data?.projects) ? data.projects : [])
      } catch (err) {
        console.error('Error loading projects:', err)
      }
    }

    fetchProjects()
  }, [user])

  if (loading) return <div className="projects-page">Loading your projects…</div>

  if (!user) {
    return (
      <div className="projects-page">
        <h1 className="projects-title">Projects</h1>
        <p className="projects-subtitle">You need to be logged in to see your projects.</p>
      </div>
    )
  }

  const createProject = async () => {
    if (!projectName.trim()) return

    const newProject = {
      id: uuidv4(),
      owner_id: user.id,
      name: projectName,
      description: projectDescription,
      created_at: new Date().toISOString(),
    }

    const response = await fetch('http://localhost:8000/add-project', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newProject),
    })

    if (!response.ok) {
      console.error('Failed to create project')
      return
    }

    // optimistic UI update
    setProjects((prev) => [newProject, ...prev])

    setProjectName('')
    setProjectDescription('')
    setShowForm(false)
  }

  return (
    <div className="projects-page">
      <button onClick={() => setShowForm(true)}>Create Project</button>

      {showForm && (
        <div className="project-form">
          <input
            type="text"
            placeholder="Project Name"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
          />

          <textarea
            placeholder="Project Description"
            value={projectDescription}
            onChange={(e) => setProjectDescription(e.target.value)}
          />

          <button onClick={createProject}>Save</button>
          <button onClick={() => setShowForm(false)}>Cancel</button>
        </div>
      )}

      <h1 className="projects-title">{user.email}&apos;s Projects</h1>

      {projects.map((p) => (
        <div key={p.id} className="project-item">
          <h2>{p.name}</h2>
          <p>{p.description}</p>
        </div>
      ))}
    </div>
  )
}
