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
    if (!response.ok) throw new Error(`Failed to fetch map data: ${response.status}`)
    const data = await response.json()
    console.log("Map data received:", {
      type: data.type,
      featureCount: data.features ? data.features.length : 0,
      sampleFeature: data.features && data.features.length > 0 ? data.features[0] : null
    })
    return data
  } catch (error) {
    console.error("Error loading map data:", error)
    return null
  }
}

function onMapClick(evt) {
    const lat = evt.latlng ? evt.latlng.lat : evt.lat;
    const lon = evt.latlng ? evt.latlng.lng : evt.lon;
    const body = { lat, lon, radius_m: 100 };

    fetch("http://localhost:8000/census_nearby", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    })
      .then((r) => r.json())
      .then((data) => {
        console.log("census_nearby:", data);
        // render popup or sidebar using data.results
        // Example: show first tract summary
        if (data.results && data.results.length) {
          const first = data.results[0];
          const html = `
            <div>
              <strong>Tract:</strong> ${first.census_tract || "n/a"}<br/>
              <strong>Parcels:</strong> ${first.parcel_count}<br/>
              <strong>Mean Median Income:</strong> ${first.mean_median_income || "n/a"}<br/>
              <strong>Mean Population:</strong> ${first.mean_population || "n/a"}
            </div>`;
          // show popup at clicked location (example using Leaflet)
          L.popup().setLatLng([lat, lon]).setContent(html).openOn(mapInstance);
        } else {
          L.popup().setLatLng([lat, lon]).setContent("No parcels found nearby").openOn(mapInstance);
        }
      })
      .catch((err) => console.error("census_nearby error", err));
}

export default function Map() {
  const [mapData, setMapData] = useState(null)
  const center = [39.9526, -75.1652] // Philadelphia
  const [showChat, setShowChat] = useState(false) 
  const [selectedPolygon, setSelectedPolygon] = useState(null)

  useEffect(() => {
    getMapData().then(data => {
      if (data && data.features && data.features.length > 0) {
        console.log(`Setting map data with ${data.features.length} features`)
        setMapData(data)
      } else {
        console.warn("No map data or empty features array:", data)
        setMapData(null)
      }
    })
  }, [])

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
        <MapContainer center={center} zoom={12} style={{ height: '100%', width: '100%' }}>
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
            <Chat plotInfo={selectedPolygon} />
          </div>
        </div>
      )}

    </div>
  )
}
