import React, { useState, useEffect } from 'react'
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet'
import Chat from './Chat'
import ParcelChat from './ParcelChat'
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
          // don't create a Leaflet popup here — just log results for now
          console.log('census_nearby result (first):', first)
        } else {
          console.log('census_nearby: no parcels found nearby')
        }
      })
      .catch((err) => console.error("census_nearby error", err))
  }

export default function Map() {
  const [mapData, setMapData] = useState(null)
  const [mapInstance, setMapInstance] = useState(null)
  const center = [39.9526, -75.17511] // Philadelphia
  const [showChat, setShowChat] = useState(false) 
  const [selectedPolygon, setSelectedPolygon] = useState(null)
  const [showParcelChat, setShowParcelChat] = useState(false)

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
        // Build a curated, user-friendly popup from properties
        const props = feature.properties || {}

        const labelMap = {
          objectid: 'Object ID',
          address: 'Address',
          owner1: 'Owner',
          bldg_desc: 'Land Type',
          opa_id: 'OPA ID',
          councildistrict: 'Council District',
          zoningbasedistrict: 'Zoning',
          zipcode: 'ZIP Code',
          land_rank: 'Land Rank',
          date_update: 'Last Update',
          Shape__Area: 'Area',
          Shape__Length: 'Perimeter'
        }

        const formatValue = (k, v) => {
          if (v === null || v === undefined || String(v).trim() === '') return null
          // Numeric formatting
          if (k === 'Shape__Area' || k === 'Shape__Length' || k === 'land_rank') {
            const num = Number(v)
            if (Number.isFinite(num)) {
              if (k === 'land_rank') return num.toFixed(2)
              return num.toLocaleString(undefined, { maximumFractionDigits: 2 })
            }
          }
          // Date formatting
          if (k === 'date_update') {
            try {
              const d = new Date(v)
              if (!isNaN(d)) return d.toLocaleDateString()
            } catch (e) { /* fall through */ }
          }
          return String(v)
        }

        const htmlParts = []
        // Order the most useful fields first
        const keysToShow = ['address','owner1','bldg_desc','zoningbasedistrict','councildistrict','zipcode','land_rank','Shape__Area','Shape__Length','date_update','opa_id','objectid']
        const used = new Set()
        for (const k of keysToShow) {
          if (props[k] !== undefined) {
            const label = labelMap[k] || k
            const val = formatValue(k, props[k])
            if (val !== null) {
              htmlParts.push(`<div style="margin-bottom:6px;"><strong>${label}:</strong> ${val}</div>`)
              used.add(k)
            }
          }
        }
        // Add any other small useful props (limit to 5), but skip noisy fields
        const skipKeys = new Set(['lniaddresskey', 'build_rank'])
        let extraCount = 0
        for (const [k,v] of Object.entries(props)) {
          if (used.has(k)) continue
          if (extraCount >= 5) break
          // Skip blacklisted keys (case-insensitive)
          if (skipKeys.has(String(k).toLowerCase())) continue
          const val = formatValue(k, v)
          if (val !== null) {
            htmlParts.push(`<div><strong>${k}:</strong> ${val}</div>`)
            extraCount += 1
          }
        }

        // Do not bind or open a Leaflet popup for features; show properties in the ParcelChat panel instead
        // const popupHtml = `<div style="font-family: sans-serif; max-width:320px">${htmlParts.join('')}</div>`
        // layer.bindPopup(popupHtml)

        layer.on("click", (e) => {
          // Prevent the global map click handler from also firing
          try { e.originalEvent && e.originalEvent.stopPropagation() } catch (err) {}
          console.log("Clicked", feature.properties)
          // highlight the clicked polygon
          layer.setStyle({ color: "red" })
          setSelectedPolygon(feature.properties)
          // show the independent parcel chat (do not open main chat)
          setShowParcelChat(true)
          setShowChat(false)
          // center and zoom the map to the parcel bounds if possible
          try {
            if (mapInstance && layer.getBounds) {
              const bounds = layer.getBounds()
              mapInstance.fitBounds(bounds, { padding: [120, 120], maxZoom: 17 })
            } else if (mapInstance && e && e.latlng) {
              mapInstance.setView([e.latlng.lat, e.latlng.lng], 16, { animate: true })
            }
          } catch (err) {
            console.warn('Could not recenter map on parcel:', err)
          }
        })

        // hover effect
        layer.on('mouseover', () => layer.setStyle({ weight: 3 }))
        layer.on('mouseout', () => layer.setStyle({ weight: 2 }))
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
          zoom={14} 
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
            {/* Pass selected polygon to Chat so it can show parcel-specific context */}
            <Chat selectedParcel={selectedPolygon} onNewChat={() => setSelectedPolygon(null)} />
          </div>
        </div>
      )}

      {/* Independent Parcel Chat (floating) */}
      {showParcelChat && selectedPolygon && (
        <ParcelChat
          parcel={selectedPolygon}
          onClose={() => setShowParcelChat(false)}
        />
      )}

    </div>
  )
}
