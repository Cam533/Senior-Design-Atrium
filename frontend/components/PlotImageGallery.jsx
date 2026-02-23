// Image upload and gallery component for plot details
import { useState, useEffect } from 'react'
import '../styles/PlotImageGallery.css'

export default function PlotImageGallery({ parcelNumber = null }) {
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedImageIndex, setSelectedImageIndex] = useState(null)

  // Debug: log when parcelNumber changes
  useEffect(() => {
    console.log('PlotImageGallery: parcelNumber =', parcelNumber)
  }, [parcelNumber])

  // Fetch images when parcelNumber changes
  useEffect(() => {
    if (!parcelNumber) {
      setImages([])
      return
    }

    const fetchImages = async () => {
      setLoading(true)
      setError(null)
      try {
        const response = await fetch(
          `http://localhost:8000/plot-images/${encodeURIComponent(parcelNumber)}`
        )
        if (!response.ok) throw new Error('Failed to fetch images')
        const data = await response.json()
        setImages(
          data.image_urls.map((url, idx) => ({
            id: data.file_ids[idx],
            url,
            loading: false,
          }))
        )
      } catch (err) {
        console.error('Error fetching images:', err)
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchImages()
  }, [parcelNumber])

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0]
    if (!file || !parcelNumber) return

    setUploading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(
        `http://localhost:8000/upload-plot-image?parcel_number=${encodeURIComponent(parcelNumber)}`,
        {
          method: 'POST',
          body: formData,
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }

      const data = await response.json()

      // Add the new image to the gallery
      setImages((prev) => [
        ...prev,
        { id: data.file_id, url: data.s3_url, loading: false },
      ])

      // Clear input
      if (event.target) event.target.value = ''
    } catch (err) {
      console.error('Error uploading image:', err)
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleDeleteImage = async (fileId) => {
    if (!parcelNumber || !window.confirm('Delete this image?')) return

    try {
      const response = await fetch(
        `http://localhost:8000/plot-image/${encodeURIComponent(parcelNumber)}/${encodeURIComponent(fileId)}`,
        { method: 'DELETE' }
      )

      if (!response.ok) throw new Error('Delete failed')

      setImages((prev) => prev.filter((img) => img.id !== fileId))
      setSelectedImageIndex(null)
    } catch (err) {
      console.error('Error deleting image:', err)
      setError(err.message)
    }
  }

  if (!parcelNumber) {
    return (
      <div className="plot-image-gallery">
        <p style={{ color: '#94a3b8' }}>No parcel selected</p>
      </div>
    )
  }

  return (
    <div className="plot-image-gallery">
      <div className="gallery-header">
        <h3>Plot Images</h3>
        <label className="upload-button">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            disabled={uploading}
            style={{ display: 'none' }}
          />
          <span>{uploading ? 'Uploading...' : '+ Add Image'}</span>
        </label>
      </div>

      {error && (
        <div className="error-message" style={{ color: '#dc2626', fontSize: 13 }}>
          {error}
        </div>
      )}

      {loading ? (
        <p style={{ color: '#94a3b8', textAlign: 'center', padding: '20px 0' }}>
          Loading images...
        </p>
      ) : images.length === 0 ? (
        <p
          style={{
            color: '#94a3b8',
            textAlign: 'center',
            padding: '20px 0',
            fontSize: 13,
          }}
        >
          No images yet. Upload one to get started.
        </p>
      ) : (
        <div className="gallery-container">
          {/* Main image display */}
          {selectedImageIndex !== null && (
            <div className="main-image-container">
              <img
                src={images[selectedImageIndex].url}
                alt={`Plot image ${selectedImageIndex + 1}`}
                className="main-image"
                onError={(e) => {
                  e.target.src =
                    'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="300"%3E%3Crect fill="%23e2e8f0" width="400" height="300"/%3E%3Ctext x="50%25" y="50%25" font-size="16" text-anchor="middle" dy=".3em" fill="%23475569"%3EImage not available%3C/text%3E%3C/svg%3E'
                }}
              />
              <div className="image-controls">
                <button
                  className="delete-btn"
                  onClick={() =>
                    handleDeleteImage(images[selectedImageIndex].id)
                  }
                  title="Delete image"
                >
                  🗑️
                </button>
                <button
                  className="close-btn"
                  onClick={() => setSelectedImageIndex(null)}
                  title="Close"
                >
                  ✕
                </button>
              </div>
            </div>
          )}

          {/* Thumbnail grid */}
          <div className="thumbnail-grid">
            {images.map((img, idx) => (
              <div
                key={img.id}
                className={`thumbnail ${selectedImageIndex === idx ? 'selected' : ''}`}
                onClick={() => setSelectedImageIndex(idx)}
              >
                <img
                  src={img.url}
                  alt={`Thumbnail ${idx + 1}`}
                  onError={(e) => {
                    e.target.src =
                      'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="80" height="80"%3E%3Crect fill="%23e2e8f0" width="80" height="80"/%3E%3C/svg%3E'
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
