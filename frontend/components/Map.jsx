import React from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
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

export default function Map() {
  const center = [39.9526, -75.1652] // Philadelphia (example)

  return (
    <div className="map-wrapper">
      <MapContainer center={center} zoom={12} style={{ height: '100%', width: '100%' }}>
        {/* Use CartoDB Positron for a clean, light basemap */}
        <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Styled circle marker instead of default blue marker */}
        {/* <CircleMarker
          center={center}
          pathOptions={{ color: '#2b8cbe', fillColor: '#2b8cbe', fillOpacity: 0.9 }}
          radius={2}
        >
          <Popup>Philadelphia center (sample)</Popup>
        </CircleMarker> */}
      </MapContainer>
    </div>
  )
}

