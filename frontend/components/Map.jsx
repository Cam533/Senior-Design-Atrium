import React, { useState, useEffect } from 'react'
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet'
import Chat from './Chat'
import L from 'leaflet'

// Fix default icon paths for Vite asset handling
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'

L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
})

const getMapData = async () => {
  try {
    const response = await fetch('http://localhost:8000/map')
    if (!response.ok) throw new Error('Failed to fetch map data')
    return await response.json()
  } catch (error) {
    console.error("Error loading map data:", error)
    return null
  }
}

const onMapClick = (evt, map) => {
    const lat = evt.latlng.lat
    const lon = evt.latlng.lng
    const body = { lat, lon, radius_m: 100 }

    fetch("http://localhost:8000/census_nearby", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    })
      .then((r) => r.json())
      .then((data) => {
        console.log("census_nearby:", data)
        if (data.results && data.results.length) {
          const first = data.results[0]
          const html = `
            <div style="font-family: sans-serif;">
              <strong>Census Tract:</strong> ${first.census_tract || "n/a"}<br/>
              <strong>Parcels in area:</strong> ${first.parcel_count}<br/>
              <strong>Median Income:</strong> $${first.mean_median_income ? Math.round(first.mean_median_income).toLocaleString() : "n/a"}<br/>
              <strong>Population:</strong> ${first.mean_population ? Math.round(first.mean_population).toLocaleString() : "n/a"}<br/>
              <strong>Median Age:</strong> ${first.mean_median_age ? Math.round(first.mean_median_age) : "n/a"}
            </div>`
          L.popup().setLatLng([lat, lon]).setContent(html).openOn(map)
        } else {
          L.popup().setLatLng([lat, lon]).setContent("No parcels found nearby").openOn(map)
        }
      })
      .catch((err) => console.error("census_nearby error", err))
  }

export default function Map() {
  const [mapData, setMapData] = useState(null)
  const [mapInstance, setMapInstance] = useState(null)
  const center = [39.9526, -75.1652] // Philadelphia
  const [showChat, setShowChat] = useState(false) 
  const [selectedPolygon, setSelectedPolygon] = useState(null)

  useEffect(() => {
    getMapData().then(data => {
      if (data) setMapData(data)
    })
  }, [])

  // Attach map click handler when mapInstance is available
  useEffect(() => {
    if (mapInstance) {
      const handler = (evt) => onMapClick(evt, mapInstance)
      mapInstance.on('click', handler)
      return () => {
        mapInstance.off('click', handler)
      }
    }
  }, [mapInstance])

  // Define the polygons layer
  const plotPolygons = mapData ? (
    <GeoJSON 
      data={mapData}
      style={{
        color: "#2b8cbe",
        weight: 2,
        fillColor: "#2b8cbe",
        fillOpacity: 0.3,
        interactive: true
      }}
      onEachFeature={(feature, layer) => {
        layer.on("click", () => {
          console.log("Clicked", feature.properties)
          // Reset style for others? (Complex with simple GeoJSON, maybe skip for now)
          layer.setStyle({ color: "red" })
          
          setSelectedPolygon(feature.properties)
          setShowChat(true)
        })
      }}
    />
  ) : null

  // need to add this somewhere: map.on('click', onMapClick);
  return (
    <div style={{ display: 'flex', height: '100%', width: '100%' }}>
      
      {/* Map Area */}
      <div style={{ flex: 1, position: 'relative' }}>
        <MapContainer 
          center={center} 
          zoom={12} 
          style={{ height: '100%', width: '100%' }}
          whenCreated={setMapInstance}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {plotPolygons}
        </MapContainer>
      </div>

      {/* Chat Sidebar */}
      {showChat && (
        <div style={{ 
          width: '400px', 
          borderLeft: '1px solid #ccc', 
          background: 'white',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{ padding: '10px', background: '#f1f5f9', display: 'flex', justifyContent: 'space-between' }}>
            <strong>Chat</strong>
            <button onClick={() => setShowChat(false)} style={{ border: 'none', background: 'transparent', cursor: 'pointer' }}>✖</button>
          </div>
          <div style={{ flex: 1, overflow: 'hidden' }}>
            {/* You can pass selectedPolygon to Chat if you want it to know context */}
            <Chat />
          </div>
        </div>
      )}

    </div>
  )
}
