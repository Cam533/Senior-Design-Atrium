import Map from '../components/Map'

export default function App() {
  return (
    <div className="app-root">
      <div className="topbar">
        <div className="brand">Atrium</div>
      </div>

      <main className="main-content">
        <div className="map-container">
          <Map />
        </div>
      </main>
    </div>
  )
}
