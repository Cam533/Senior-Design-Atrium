import { useState, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { useAuth } from '../src/context/AuthContext'
import Map from './Map'
import LotDetails from './LotDetails'
import '../styles/Projects.css'

// Unique id for a parcel (for dedup and API)
function getParcelId(p) {
  return p?.objectid ?? p?.parcel_number ?? p?.parcelNumber ?? p?.opa_id ?? p?.address ?? `${p?.lat ?? ''},${p?.lon ?? ''}` ?? ''
}

export default function Projects() {
  const { user, loading } = useAuth()

  const [projects, setProjects] = useState([]) // start empty
  const [showForm, setShowForm] = useState(false)
  const [projectName, setProjectName] = useState('')
  const [projectDescription, setProjectDescription] = useState('')
  const [projectPlots, setProjectPlots] = useState([]) // array of parcel objects from map
  const [showMapPicker, setShowMapPicker] = useState(false)
  const [expandedProjectId, setExpandedProjectId] = useState(null) // which project's plots are expanded
  const [selectedParcel, setSelectedParcel] = useState(null) // for LotDetails
  const [plotsCache, setPlotsCache] = useState({}) // projectId -> { loading, parcels }
  const [editingProjectId, setEditingProjectId] = useState(null)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [savingEdit, setSavingEdit] = useState(false)
  const [editingPlotsProjectId, setEditingPlotsProjectId] = useState(null)

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

  if (loading) {
    return (
      <div className="projects-page">
        <div className="projects-loading">Loading your projects…</div>
      </div>
    )
  }

  if (!user) {
    return (
      <div className="projects-page">
        <div className="projects-login-prompt">
          <h1 className="projects-title">Projects</h1>
          <p className="projects-subtitle">You must be logged in to see your projects.</p>
        </div>
      </div>
    )
  }

  if (selectedParcel) {
    return (
      <LotDetails
        parcel={selectedParcel}
        onBack={() => setSelectedParcel(null)}
      />
    )
  }

  const createProject = async () => {
    if (!projectName.trim()) return

    const plotIds = Array.isArray(projectPlots)
      ? projectPlots.map((p) => String(getParcelId(p))).filter(Boolean)
      : []

    const newProject = {
      id: uuidv4(),
      owner_id: user.id,
      name: projectName.trim(),
      description: typeof projectDescription === 'string' ? projectDescription : '',
      plots: plotIds,
      created_at: new Date().toISOString(),
    }

    const response = await fetch('http://localhost:8000/add-project', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newProject),
    })

    if (!response.ok) {
      console.error('Failed to create project:', response.statusText)
      return
    }

    // optimistic UI update
    setProjects((prev) => [newProject, ...prev])

    setProjectName('')
    setProjectDescription('')
    setProjectPlots([])
    setShowMapPicker(false)
    setShowForm(false)
  }

  const addPlotFromMap = (parcel) => {
    const id = getParcelId(parcel)
    if (!id) return
    setProjectPlots((prev) => {
      if (prev.some((p) => getParcelId(p) === id)) return prev
      return [...prev, parcel]
    })
  }

  const removePlot = (parcel) => {
    const id = getParcelId(parcel)
    setProjectPlots((prev) => prev.filter((p) => getParcelId(p) !== id))
  }

  const formatDate = (iso) => {
    try {
      const d = new Date(iso)
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
    } catch {
      return ''
    }
  }

  const fetchProjectPlots = async (projectId, objectids) => {
    if (!objectids?.length) {
      setPlotsCache((c) => ({ ...c, [projectId]: { loading: false, parcels: [] } }))
      return
    }
    setPlotsCache((c) => ({ ...c, [projectId]: { ...c[projectId], loading: true } }))
    try {
      const res = await fetch('http://localhost:8000/parcels_by_ids', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ objectids: objectids.map(String) }),
      })
      const data = await res.json()
      const parcels = Array.isArray(data?.parcels) ? data.parcels : []
      setPlotsCache((c) => ({ ...c, [projectId]: { loading: false, parcels } }))
    } catch (err) {
      console.error('Failed to fetch project plots:', err)
      setPlotsCache((c) => ({ ...c, [projectId]: { loading: false, parcels: [] } }))
    }
  }

  const toggleShowPlots = (project) => {
    const id = project.id
    const isExpanding = expandedProjectId !== id
    setExpandedProjectId(isExpanding ? id : null)
    if (isExpanding && project.plots?.length && !plotsCache[id]?.parcels && !plotsCache[id]?.loading) {
      fetchProjectPlots(id, project.plots)
    }
  }

  const startEditingProject = (project) => {
    setEditingProjectId(project.id)
    setEditName(project.name || '')
    setEditDescription(project.description || '')
  }

  const cancelEditingProject = () => {
    setEditingProjectId(null)
    setEditName('')
    setEditDescription('')
    setSavingEdit(false)
  }

  const saveProjectEdits = async (project) => {
    if (!editName.trim() || !user?.id || savingEdit) return
    setSavingEdit(true)
    try {
      const payload = {
        id: project.id,
        owner_id: user.id,
        name: editName.trim(),
        description: editDescription,
        plots: Array.isArray(project.plots) ? project.plots : [],
      }
      const response = await fetch('http://localhost:8000/update-project', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        console.error('Failed to update project:', response.statusText)
        setSavingEdit(false)
        return
      }
      setProjects((prev) =>
        prev.map((p) =>
          p.id === project.id
            ? { ...p, name: payload.name, description: payload.description }
            : p
        )
      )
      cancelEditingProject()
    } catch (err) {
      console.error('Error updating project:', err)
      setSavingEdit(false)
    }
  }

  const startEditingProjectPlots = async (project) => {
    const projectId = project.id
    setEditingPlotsProjectId(projectId)
    setShowForm(true)
    setShowMapPicker(true)
    setProjectName(project.name || '')
    setProjectDescription(project.description || '')

    const cachedParcels = plotsCache[projectId]?.parcels
    if (Array.isArray(cachedParcels)) {
      setProjectPlots(cachedParcels)
      return
    }
    const plotIds = Array.isArray(project.plots) ? project.plots : []
    if (!plotIds.length) {
      setProjectPlots([])
      return
    }
    try {
      const res = await fetch('http://localhost:8000/parcels_by_ids', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ objectids: plotIds.map(String) }),
      })
      const data = await res.json()
      const parcels = Array.isArray(data?.parcels) ? data.parcels : []
      setProjectPlots(parcels)
      setPlotsCache((c) => ({ ...c, [projectId]: { loading: false, parcels } }))
    } catch (err) {
      console.error('Failed to load plots for editing:', err)
      setProjectPlots([])
    }
  }

  const saveEditedProjectPlots = async () => {
    if (!editingPlotsProjectId || !user?.id) return
    const project = projects.find((p) => p.id === editingPlotsProjectId)
    if (!project) return

    const nextName = (typeof projectName === 'string' ? projectName : '').trim()
    if (!nextName) return
    const nextDescription = typeof projectDescription === 'string' ? projectDescription : ''

    const plotIds = Array.isArray(projectPlots)
      ? projectPlots.map((parcel) => String(getParcelId(parcel))).filter(Boolean)
      : []

    const payload = {
      id: project.id,
      owner_id: user.id,
      name: nextName,
      description: nextDescription,
      plots: plotIds,
    }

    try {
      const response = await fetch('http://localhost:8000/update-project', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        console.error('Failed to save project plots:', response.statusText)
        return
      }
      setProjects((prev) =>
        prev.map((p) =>
          p.id === project.id
            ? { ...p, name: nextName, description: nextDescription, plots: plotIds }
            : p
        )
      )
      setPlotsCache((c) => ({
        ...c,
        [project.id]: { loading: false, parcels: projectPlots },
      }))
      setShowMapPicker(false)
      setShowForm(false)
      setEditingPlotsProjectId(null)
      setProjectPlots([])
      setProjectName('')
      setProjectDescription('')
    } catch (err) {
      console.error('Error saving project plots:', err)
    }
  }

  return (
    <div className={`projects-page ${showMapPicker ? 'projects-page-with-map' : ''}`}>
      <header className="projects-header">
        <div>
          <h1 className="projects-title">Projects</h1>
          <p className="projects-subtitle">Create and manage your development projects.</p>
        </div>
        {!showForm && (
          <button
            type="button"
            className="projects-create-btn"
            onClick={() => setShowForm(true)}
          >
            + Create Project
          </button>
        )}
      </header>

      {showForm && (
        <div className="projects-form-layout">
          <div className="project-form-wrapper">
            <div className="project-form">
              <h2 className="project-form-title">New project</h2>
              {editingPlotsProjectId && (
                <p className="projects-subtitle" style={{ margin: '0 0 12px 0' }}>
                  Editing plots for an existing project. Save to update its plot list.
                </p>
              )}
              <label htmlFor="project-name">Project name</label>
              <input
                id="project-name"
                type="text"
                placeholder="e.g. North Philly mixed-use"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                autoFocus
              />
              <label htmlFor="project-description">Description (optional)</label>
              <textarea
                id="project-description"
                placeholder="Brief description of the project…"
                value={projectDescription}
                onChange={(e) => setProjectDescription(e.target.value)}
              />
              <div className="project-form-plots">
                <label>Plots (optional)</label>
                <p className="project-form-plots-hint">Click parcels on the map to add them to this project.</p>
                <button
                  type="button"
                  className="project-form-open-map"
                  onClick={() => setShowMapPicker((v) => !v)}
                >
                  {showMapPicker ? 'Hide map' : 'Open map to pick plots'}
                </button>
                {projectPlots.length > 0 && (
                  <ul className="project-plots-list">
                    {projectPlots.map((p) => (
                      <li key={getParcelId(p)} className="project-plot-chip">
                        <span className="project-plot-chip-label">
                          {p.address || p.objectid || getParcelId(p)}
                        </span>
                        <button
                          type="button"
                          className="project-plot-chip-remove"
                          onClick={() => removePlot(p)}
                          aria-label="Remove plot"
                        >
                          ×
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              <div className="project-form-actions">
                <button
                  type="button"
                  className="project-form-save"
                  onClick={editingPlotsProjectId ? saveEditedProjectPlots : createProject}
                  disabled={!projectName.trim() && !editingPlotsProjectId}
                >
                  {editingPlotsProjectId ? 'Save plot changes' : 'Save project'}
                </button>
                <button
                  type="button"
                  className="project-form-cancel"
                  onClick={() => {
                    setShowForm(false)
                    setProjectName('')
                    setProjectDescription('')
                    setProjectPlots([])
                    setShowMapPicker(false)
                    setEditingPlotsProjectId(null)
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>

          {showMapPicker && (
            <div className="projects-map-panel">
              <div className="projects-map-panel-header">
                <span>Click a parcel to add it to the project</span>
                <button
                  type="button"
                  className="project-form-map-close"
                  onClick={() => setShowMapPicker(false)}
                  aria-label="Close map"
                >
                  ×
                </button>
              </div>
              <div className="projects-map-container">
                <Map onParcelSelectForProject={addPlotFromMap} />
              </div>
            </div>
          )}
        </div>
      )}

      <h2 className="projects-list-title">Your projects</h2>

      {projects.length === 0 ? (
        <div className="projects-empty">
          <strong>No projects yet</strong>
          Click “Create Project” to add your first one.
        </div>
      ) : (
        <div className="projects-list">
          {projects.map((p) => (
            <article key={p.id} className="project-item">
              <div className="project-item-header">
                {editingProjectId === p.id ? (
                  <div className="project-item-edit-fields">
                    <input
                      type="text"
                      className="project-item-edit-name"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      placeholder="Project name"
                    />
                    <textarea
                      className="project-item-edit-description"
                      value={editDescription}
                      onChange={(e) => setEditDescription(e.target.value)}
                      placeholder="Description (optional)"
                    />
                  </div>
                ) : (
                  <div>
                    <h3 className="project-item-name">{p.name}</h3>
                    {p.description ? (
                      <p className="project-item-description">{p.description}</p>
                    ) : (
                      <p className="project-item-description" aria-hidden>No description.</p>
                    )}
                  </div>
                )}
                {editingProjectId === p.id ? (
                  <div className="project-item-edit-actions">
                    <button
                      type="button"
                      className="project-item-save-btn"
                      onClick={() => saveProjectEdits(p)}
                      disabled={!editName.trim() || savingEdit}
                    >
                      {savingEdit ? 'Saving…' : 'Save'}
                    </button>
                    <button
                      type="button"
                      className="project-item-cancel-btn"
                      onClick={cancelEditingProject}
                      disabled={savingEdit}
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="project-item-edit-btn"
                    onClick={() => startEditingProjectPlots(p)}
                  >
                    Edit
                  </button>
                )}
              </div>
              {p.plots && p.plots.length > 0 && (
                <div className="project-item-plots">
                  <button
                    type="button"
                    className="project-item-show-plots-btn"
                    onClick={() => toggleShowPlots(p)}
                  >
                    {expandedProjectId === p.id ? 'Hide saved plots' : 'Show saved plots'}
                  </button>
                  {expandedProjectId === p.id && (
                    <div className="project-item-plots-list">
                      {plotsCache[p.id]?.loading ? (
                        <p className="project-item-plots-loading">Loading plots…</p>
                      ) : (plotsCache[p.id]?.parcels?.length ? (
                        <ul className="project-plots-saved-list">
                          {(plotsCache[p.id].parcels).map((parcel) => (
                            <li key={getParcelId(parcel)} className="project-plot-saved-row">
                              <div>
                                <div className="project-plot-saved-address">
                                  {parcel.address || parcel.objectid || getParcelId(parcel)}
                                </div>
                                {parcel.zoningbasedistrict && (
                                  <div className="project-plot-saved-zoning">
                                    Zoning: {parcel.zoningbasedistrict}
                                  </div>
                                )}
                              </div>
                              <button
                                type="button"
                                className="project-plot-view-details-btn"
                                onClick={() => setSelectedParcel(parcel)}
                              >
                                View Details
                              </button>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="project-item-plots-empty">No parcel data found for these plot IDs.</p>
                      ))}

                    </div>
                  )}
                </div>
              )}
              {p.created_at && (
                <p className="project-item-meta">
                  Created {formatDate(p.created_at)}
                </p>
              )}
            </article>
          ))}
        </div>
      )}
    </div>
  )
}
