import React, { useState, useEffect, useRef } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import { Popup, CircleMarker } from "react-leaflet";
import Chat from "./Chat";
import ParcelChat from "./ParcelChat";
import L from "leaflet";

// Fix default icon paths for Vite asset handling
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

const getMapData = async () => {
  try {
    const response = await fetch("http://localhost:8000/map");
    if (!response.ok) throw new Error("Failed to fetch map data");
    return await response.json();
  } catch (error) {
    console.error("Error loading map data:", error);
    return null;
  }
};

const onMapClick = (evt, map) => {
  const lat = evt.latlng.lat;
  const lon = evt.latlng.lng;
  const body = { lat, lon, radius_m: 100 };

  fetch("http://localhost:8000/census_nearby", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then((r) => r.json())
    .then((data) => {
      console.log("census_nearby:", data);
      if (data.results && data.results.length) {
        const first = data.results[0];
        // don't create a Leaflet popup here — just log results for now
        console.log("census_nearby result (first):", first);
      } else {
        console.log("census_nearby: no parcels found nearby");
      }
    })
    .catch((err) => console.error("census_nearby error", err));
};

// ChunkedGeoJSON component: progressively add GeoJSON features in small batches
function ChunkedGeoJSON({ data, batchSize = 300, options = {} }) {
  const map = useMap();
  const layerRef = useRef(null);

  useEffect(() => {
    if (!data || !data.features || !map) return;

    // create an empty geoJSON layer with provided options
    const g = L.geoJSON(null, options).addTo(map);
    layerRef.current = g;

    let idx = 0;
    const features = data.features || [];

    function addBatch() {
      const start = idx;
      const end = Math.min(idx + batchSize, features.length);
      for (let i = start; i < end; i++) {
        try {
          g.addData(features[i]);
        } catch (err) {
          // skip malformed features
          console.warn("Failed to add feature", err);
        }
      }
      idx = end;
      if (idx < features.length) {
        if (window.requestIdleCallback) {
          window.requestIdleCallback(addBatch, { timeout: 200 });
        } else {
          setTimeout(addBatch, 16);
        }
      }
    }

    addBatch();

    return () => {
      try {
        map.removeLayer(g);
      } catch (err) {}
    };
    // Intentionally keep options out of deps here; options should be memoized
  }, [data, map, batchSize]);

  return null;
}

// Renders circle markers for nearest park and transit on the map
function NearestPlacesMarkers({ nearestPark, nearestTransitStop }) {
  const hasValidNearest = (place) =>
    place &&
    typeof place.lat === "number" &&
    typeof place.lon === "number" &&
    Number.isFinite(Number(place.distance_m));

  return (
    <>
      {hasValidNearest(nearestPark) && (
        <CircleMarker
          center={[nearestPark.lat, nearestPark.lon]}
          pathOptions={{ color: "#16a34a", fillColor: "#16a34a", fillOpacity: 0.8, weight: 2 }}
          radius={10}
        >
          <Popup>
            <strong>Nearest park</strong>
            {nearestPark.name && <div>{nearestPark.name}</div>}
            {nearestPark.address && <div>{nearestPark.address}</div>}
            {nearestPark.distance_m != null && (
              <div>{nearestPark.distance_m < 1000 ? `${Math.round(nearestPark.distance_m)} m` : `${(nearestPark.distance_m / 1000).toFixed(1)} km`}</div>
            )}
          </Popup>
        </CircleMarker>
      )}
      {hasValidNearest(nearestTransitStop) && (
        <CircleMarker
          center={[nearestTransitStop.lat, nearestTransitStop.lon]}
          pathOptions={{ color: "#2563eb", fillColor: "#2563eb", fillOpacity: 0.8, weight: 2 }}
          radius={10}
        >
          <Popup>
            <strong>Nearest transit</strong>
            {nearestTransitStop.name && <div>{nearestTransitStop.name}</div>}
            {nearestTransitStop.address && <div>{nearestTransitStop.address}</div>}
            {nearestTransitStop.distance_m != null && (
              <div>{nearestTransitStop.distance_m < 1000 ? `${Math.round(nearestTransitStop.distance_m)} m` : `${(nearestTransitStop.distance_m / 1000).toFixed(1)} km`}</div>
            )}
          </Popup>
        </CircleMarker>
      )}
    </>
  );
}

export default function Map({ onParcelSelectForProject = null }) {
  const [mapData, setMapData] = useState(null);
  const [mapInstance, setMapInstance] = useState(null);
  const selectedLayerRef = useRef(null);
  const center = [39.9526, -75.17511]; // Philadelphia
  const [showChat, setShowChat] = useState(false);
  const [selectedPolygon, setSelectedPolygon] = useState(null);
  const [showParcelChat, setShowParcelChat] = useState(false);
  const [showFilterModal, setShowFilterModal] = useState(false);
  const [nearestPark, setNearestPark] = useState(null);
  const [nearestTransitStop, setNearestTransitStop] = useState(null);
  const [filters, setFilters] = useState({
    landTypes: [],
    councilDistricts: [],
    zoningDistricts: [],
    zipCodes: [],
    owners: [],
    minLandRank: null,
    maxLandRank: null,
    minShapeArea: null,
    maxShapeArea: null,
    minEnvironmentalScore: null,
    maxEnvironmentalScore: null,
    minRecreationalScore: null,
    maxRecreationalScore: null,
    minTransitScore: null,
    maxTransitScore: null,
    minWalkabilityScore: null,
    maxWalkabilityScore: null,
  });

  const defaultLayerStyle = React.useMemo(() => ({
    color: "#2b8cbe",
    weight: 2,
    fillColor: "#2b8cbe",
    fillOpacity: 0.3,
    interactive: true,
  }), []);

  useEffect(() => {
    getMapData().then((data) => {
      if (data) setMapData(data);
    });
  }, []);

  // Attach map click handler when mapInstance is available
  useEffect(() => {
    if (mapInstance) {
      const handler = (evt) => onMapClick(evt, mapInstance);
      mapInstance.on("click", handler);
      return () => {
        mapInstance.off("click", handler);
      };
    }
  }, [mapInstance]);

  // display nearest park and transit stop on map

  // Define the per-feature interaction handler so it can be used by the chunked loader
  const handleEachFeature = React.useCallback((feature, layer) => {
    const props = feature.properties || {};

    const labelMap = {
      address: "Address",
      owner1: "Owner",
      bldg_desc: "Land Type",
      councildistrict: "Council District",
      zoningbasedistrict: "Zoning",
      zipcode: "ZIP Code",
      land_rank: "Land Rank",
      date_update: "Last Update",
    };

    const formatValue = (k, v) => {
      if (v === null || v === undefined || String(v).trim() === "") return null;
      if (k === "Shape__Area" || k === "Shape__Length" || k === "land_rank") {
        const num = Number(v);
        if (Number.isFinite(num)) {
          if (k === "land_rank") return num.toFixed(2);
          return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
        }
      }
      if (k === "date_update") {
        try {
          const d = new Date(v);
          if (!isNaN(d)) return d.toLocaleDateString();
        } catch (e) {}
      }
      return String(v);
    };

    const htmlParts = [];
    const keysToShow = [
      "address",
      "owner1",
      "bldg_desc",
      "zoningbasedistrict",
      "councildistrict",
      "zipcode",
      "land_rank",
      "date_update",
    ];
    const used = new Set();
    for (const k of keysToShow) {
      if (props[k] !== undefined) {
        const label = labelMap[k] || k;
        const val = formatValue(k, props[k]);
        if (val !== null) {
          htmlParts.push(`<div style="margin-bottom:6px;"><strong>${label}:</strong> ${val}</div>`);
          used.add(k);
        }
      }
    }
    const skipKeys = new Set(["lniaddresskey", "build_rank", "objectid", "opa_id", "shape__area", "shape__length", "lat", "lon"]);
    let extraCount = 0;
    for (const [k, v] of Object.entries(props)) {
      if (used.has(k)) continue;
      if (extraCount >= 5) break;
      if (skipKeys.has(String(k).toLowerCase())) continue;
      const val = formatValue(k, v);
      if (val !== null) {
        const label = labelMap[k] || String(k).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        htmlParts.push(`<div><strong>${label}:</strong> ${val}</div>`);
        extraCount += 1;
      }
    }

    layer.on("click", (e) => {
      try {
        e.originalEvent && e.originalEvent.stopPropagation();
      } catch (err) {}
      console.log("Clicked", feature.properties);

      let lat = null;
      let lon = null;

      if (feature.geometry && feature.geometry.coordinates) {
        const coords = feature.geometry.coordinates;
        if (feature.geometry.type === "Polygon" && coords[0] && coords[0].length > 0) {
          const polygonCoords = coords[0];
          let sumLat = 0,
            sumLon = 0;
          for (const coord of polygonCoords) {
            sumLon += coord[0];
            sumLat += coord[1];
          }
          lon = sumLon / polygonCoords.length;
          lat = sumLat / polygonCoords.length;
        } else if (feature.geometry.type === "MultiPolygon") {
          const firstPolygon = coords[0];
          if (firstPolygon && firstPolygon[0] && firstPolygon[0].length > 0) {
            const polygonCoords = firstPolygon[0];
            let sumLat = 0,
              sumLon = 0;
            for (const coord of polygonCoords) {
              sumLon += coord[0];
              sumLat += coord[1];
            }
            lon = sumLon / polygonCoords.length;
            lat = sumLat / polygonCoords.length;
          }
        }
      }

      const parcelWithCoords = {
        ...feature.properties,
        lat: lat || e.latlng.lat,
        lon: lon || e.latlng.lng,
      };

      // reset previous selection style (if a different layer)
      try {
        if (selectedLayerRef.current && selectedLayerRef.current !== layer) {
          selectedLayerRef.current.setStyle && selectedLayerRef.current.setStyle(defaultLayerStyle);
        }
      } catch (err) {
        /* ignore */
      }

      // highlight this layer and remember it
      layer.setStyle({ color: "red" });
      selectedLayerRef.current = layer;

      // In project mode, still add the parcel to the project list
      // but continue to open the same parcel info/chat UI as the main map.
      if (onParcelSelectForProject && typeof onParcelSelectForProject === "function") {
        onParcelSelectForProject(parcelWithCoords);
      }

      setSelectedPolygon(parcelWithCoords);
      // show the independent parcel chat (do not open main chat)
      setShowParcelChat(true);
      setShowChat(false);

      // Animate zoom/pan to the parcel and offset it so it appears
      // left-of-center when the chat opens (leaves room on the right).
      // Use the layer's map reference when possible to avoid stale closures
      try {
        const map = (layer && layer._map) || mapInstance;
        if (map) {
          // Determine a target latlng and zoom. Prefer bounds if available.
          let targetLatLng = null;
          let targetZoom = 16;
          if (layer.getBounds && typeof layer.getBounds === "function") {
            const bounds = layer.getBounds();
            targetLatLng = bounds.getCenter();
            // try to compute a zoom that fits the bounds (fallback to +2 zoom)
            try {
              if (typeof map.getBoundsZoom === "function") {
                const z = map.getBoundsZoom(bounds, false);
                if (Number.isFinite(z)) targetZoom = Math.max(z, 17);
              } else {
                targetZoom = Math.max(map.getZoom(), 16);
              }
            } catch (err) {
              targetZoom = Math.max(map.getZoom(), 16);
            }
          }

          if (!targetLatLng && e && e.latlng) {
            targetLatLng = e.latlng;
          }

          // If we have a target, fly to the parcel's geographic center (centered)
          if (targetLatLng) {
            const centerOffset = L.point(15, 0); // offset left by 200 pixels
            const latLng = map.containerPointToLatLng(
              map.latLngToContainerPoint(targetLatLng).add(centerOffset)
            );
            map.flyTo(latLng, targetZoom, { animate: true, duration: 0.8 });
          }
        }
      } catch (err) {
        console.warn("Could not recenter map on parcel:", err);
      }
    });

    layer.on("mouseover", () => layer.setStyle({ weight: 3 }));
    layer.on("mouseout", () => layer.setStyle({ weight: 2 }));
  }, [setSelectedPolygon, setShowParcelChat, setShowChat, defaultLayerStyle, onParcelSelectForProject]);

  // Create a memoized options object so the chunked loader doesn't restart
  const chunkOptions = React.useMemo(() => {
    return {
      style: {
        color: "#2b8cbe",
        weight: 2,
        fillColor: "#2b8cbe",
        fillOpacity: 0.3,
        interactive: true,
      },
      onEachFeature: handleEachFeature,
      renderer: L.canvas({ tolerance: 10 }),
    };
  }, [handleEachFeature]);

  // When the parcel chat closes or the selection is cleared, reset the previous layer's style
  useEffect(() => {
    if (!showParcelChat || !selectedPolygon) {
      if (selectedLayerRef.current) {
        try {
          selectedLayerRef.current.setStyle && selectedLayerRef.current.setStyle(defaultLayerStyle);
        } catch (err) {}
        selectedLayerRef.current = null;
      }
      setNearestPark(null);
      setNearestTransitStop(null);
    }
  }, [showParcelChat, selectedPolygon, defaultLayerStyle]);

  // Fetch nearest park and transit for the selected parcel so we can show them on the map
  useEffect(() => {
    const lat = selectedPolygon?.lat;
    const lon = selectedPolygon?.lon;
    if (lat == null || lon == null || typeof lat !== "number" || typeof lon !== "number") return;
    let cancelled = false;
    fetch("http://localhost:8000/geographic_scores", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (cancelled) return;
        const park = data?.nearest_park;
        const transit = data?.nearest_transit_stop;
        if (
          park &&
          typeof park.lat === "number" &&
          typeof park.lon === "number" &&
          Number.isFinite(Number(park.distance_m))
        ) setNearestPark(park);
        else setNearestPark(null);
        if (
          transit &&
          typeof transit.lat === "number" &&
          typeof transit.lon === "number" &&
          Number.isFinite(Number(transit.distance_m))
        ) setNearestTransitStop(transit);
        else setNearestTransitStop(null);
      })
      .catch(() => {
        if (!cancelled) {
          setNearestPark(null);
          setNearestTransitStop(null);
        }
      });
    return () => { cancelled = true; };
  }, [selectedPolygon?.lat, selectedPolygon?.lon]);

  // Extract unique filter values from mapData
  const filterOptions = React.useMemo(() => {
    if (!mapData || !mapData.features) return { landTypes: [], councilDistricts: [], zoningDistricts: [], zipCodes: [], owners: [] };
    
    const landTypes = new Set();
    const councilDistricts = new Set();
    const zoningDistricts = new Set();
    const zipCodes = new Set();
    const owners = new Set();
    
    mapData.features.forEach((feature) => {
      const props = feature.properties || {};
      if (props.bldg_desc) landTypes.add(props.bldg_desc);
      if (props.councildistrict) councilDistricts.add(String(props.councildistrict));
      if (props.zoningbasedistrict) zoningDistricts.add(props.zoningbasedistrict);
      if (props.zipcode) zipCodes.add(String(props.zipcode));
      const owner = props.owner1;
      if (owner != null && String(owner).trim() !== "") owners.add(String(owner).trim());
    });
    
    return {
      landTypes: Array.from(landTypes).sort(),
      councilDistricts: Array.from(councilDistricts).sort((a, b) => Number(a) - Number(b)),
      zoningDistricts: Array.from(zoningDistricts).sort(),
      zipCodes: Array.from(zipCodes).sort(),
      owners: Array.from(owners).sort(),
    };
  }, [mapData]);

  // Filter mapData based on selected filters
  const filteredMapData = React.useMemo(() => {
    if (!mapData || !mapData.features) return mapData;
    
    // Check if any filters are active
    const hasActiveFilters = 
      filters.landTypes.length > 0 ||
      filters.councilDistricts.length > 0 ||
      filters.zoningDistricts.length > 0 ||
      filters.zipCodes.length > 0 ||
      filters.owners.length > 0 ||
      filters.minLandRank !== null ||
      filters.maxLandRank !== null ||
      filters.minShapeArea !== null ||
      filters.maxShapeArea !== null ||
      filters.minEnvironmentalScore !== null ||
      filters.maxEnvironmentalScore !== null ||
      filters.minRecreationalScore !== null ||
      filters.maxRecreationalScore !== null ||
      filters.minTransitScore !== null ||
      filters.maxTransitScore !== null ||
      filters.minWalkabilityScore !== null ||
      filters.maxWalkabilityScore !== null;
    
    // If no filters are active, return all data
    if (!hasActiveFilters) {
      return mapData;
    }
    
    // Filter features based on criteria
    const filteredFeatures = mapData.features.filter((feature) => {
      const props = feature.properties || {};
      
      // Check land types filter
      if (filters.landTypes.length > 0) {
        if (!filters.landTypes.includes(props.bldg_desc)) {
          return false;
        }
      }
      
      // Check council districts filter
      if (filters.councilDistricts.length > 0) {
        const district = String(props.councildistrict || '');
        if (!filters.councilDistricts.includes(district)) {
          return false;
        }
      }
      
      // Check zoning districts filter
      if (filters.zoningDistricts.length > 0) {
        if (!filters.zoningDistricts.includes(props.zoningbasedistrict)) {
          return false;
        }
      }
      
      // Check ZIP codes filter
      if (filters.zipCodes.length > 0) {
        const zip = String(props.zipcode || '');
        if (!filters.zipCodes.includes(zip)) {
          return false;
        }
      }
      
      // Check owner filter (owner1)
      if (filters.owners.length > 0) {
        const ownerVal = props.owner1 != null ? String(props.owner1).trim() : "";
        if (!ownerVal || !filters.owners.includes(ownerVal)) {
          return false;
        }
      }
      
      // Check land rank range filter
      if (filters.minLandRank !== null || filters.maxLandRank !== null) {
        const landRank = Number(props.land_rank);
        if (isNaN(landRank)) {
          return false; // Exclude if land_rank is not a number
        }
        if (filters.minLandRank !== null && landRank < filters.minLandRank) {
          return false;
        }
        if (filters.maxLandRank !== null && landRank > filters.maxLandRank) {
          return false;
        }
      }

      // Check Shape__Area range filter
      if (filters.minShapeArea !== null || filters.maxShapeArea !== null) {
        const shapeArea = Number(props.Shape__Area);
        if (!Number.isFinite(shapeArea)) return false;
        if (filters.minShapeArea !== null && shapeArea < filters.minShapeArea) return false;
        if (filters.maxShapeArea !== null && shapeArea > filters.maxShapeArea) return false;
      }

      // Helper to apply a score range filter if present.
      // If the feature doesn't have the score, exclude it when that filter is active.
      const scoreInRange = (raw, min, max) => {
        if (min === null && max === null) return true;
        const v = Number(raw);
        if (!Number.isFinite(v)) return false;
        if (min !== null && v < min) return false;
        if (max !== null && v > max) return false;
        return true;
      };

      if (!scoreInRange(props.environmental_score, filters.minEnvironmentalScore, filters.maxEnvironmentalScore)) return false;
      if (!scoreInRange(props.recreational_score, filters.minRecreationalScore, filters.maxRecreationalScore)) return false;
      if (!scoreInRange(props.transit_score, filters.minTransitScore, filters.maxTransitScore)) return false;
      if (!scoreInRange(props.walkability_score, filters.minWalkabilityScore, filters.maxWalkabilityScore)) return false;
      
      return true;
    });
    
    return {
      ...mapData,
      features: filteredFeatures,
    };
  }, [mapData, filters]);

  // Zoom back out a bit when the parcel chat closes (but only when it was open)
  const prevShowParcelChat = useRef(showParcelChat);
  useEffect(() => {
    if (prevShowParcelChat.current && !showParcelChat && mapInstance) {
      try {
        const map = mapInstance;
        const currentZoom = map.getZoom();
        const newZoom = Math.max(10, currentZoom - 1); // zoom out by 1, but not too far
        const center = map.getCenter();
        const centerOffset = L.point(-15, 0); // offset left by 200 pixels
        const targetLatLng = map.containerPointToLatLng(
          map.latLngToContainerPoint(center).add(centerOffset)
        );
        map.flyTo(targetLatLng, newZoom, { animate: true, duration: 0.6 });
      } catch (err) {
        console.warn("Failed to zoom out on chat close:", err);
      }
    }
    prevShowParcelChat.current = showParcelChat;
  }, [showParcelChat, mapInstance]);

  // need to add this somewhere: map.on('click', onMapClick);
  return (
    <div style={{ display: "flex", height: "100%", width: "100%" }}>
      {/* Map Area */}
      <div style={{ flex: 1, position: "relative" }}>
        {/* Placeholder Button - Left side, top of map */}
        <button
          style={{
            position: "absolute",
            top: "100px",
            left: "10px",
            zIndex: 1000,
            padding: "10px 20px",
            backgroundColor: "#fff",
            border: "1px solid #ccc",
            borderRadius: "4px",
            cursor: "pointer",
            boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
            fontWeight: "bold",
          }}
          onClick={() => {
            // Placeholder onClick handler
            setShowFilterModal(true);
            console.log("Filter button clicked");
          }}
        >
          Filter
        </button>
        <MapContainer
          center={center}
          zoom={14}
          style={{ height: "100%", width: "100%" }}
          whenCreated={setMapInstance}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {filteredMapData && <ChunkedGeoJSON data={filteredMapData} batchSize={1000} options={chunkOptions} />}
          <NearestPlacesMarkers nearestPark={nearestPark} nearestTransitStop={nearestTransitStop} />
        </MapContainer>
      </div>

      {/* Chat Sidebar */}
      {showChat && (
        <div
          style={{
            width: "400px",
            borderLeft: "1px solid #ccc",
            background: "white",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              padding: "10px",
              background: "#f1f5f9",
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <strong>Chat</strong>
            <button
              onClick={() => setShowChat(false)}
              style={{
                border: "none",
                background: "transparent",
                cursor: "pointer",
              }}
            >
              ✖
            </button>
          </div>
          <div style={{ flex: 1, overflow: "hidden" }}>
            {/* Pass selected polygon to Chat so it can show parcel-specific context */}
            <Chat
              selectedParcel={selectedPolygon}
              onNewChat={() => setSelectedPolygon(null)}
            />
          </div>
        </div>
      )}
      {selectedPolygon && !showParcelChat && (
        <button
          onClick={() => setShowParcelChat(true)}
          style={{
            position: "absolute",
            top: 100,
            right: 16,
            zIndex: 1200,

            display: "inline-flex",
            alignItems: "center",
            gap: 8,

            padding: "10px 14px",
            borderRadius: 9999,

            background: "white",
            color: "#111827",
            border: "1px solid rgba(0,0,0,0.12)",
            boxShadow: "0 8px 20px rgba(0,0,0,0.12)",

            fontSize: 14,
            fontWeight: 600,
            letterSpacing: 0.2,

            cursor: "pointer",
            userSelect: "none",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.boxShadow = "0 10px 24px rgba(0,0,0,0.16)"
            e.currentTarget.style.transform = "translateY(-1px)"
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.boxShadow = "0 8px 20px rgba(0,0,0,0.12)"
            e.currentTarget.style.transform = "translateY(0px)"
          }}
          onMouseDown={(e) => {
            e.currentTarget.style.transform = "translateY(0px) scale(0.98)"
          }}
          onMouseUp={(e) => {
            e.currentTarget.style.transform = "translateY(-1px)"
          }}
        >
          Open Chat
        </button>
      )}



      {/* Independent Parcel Chat (button to open and close) */}
      {showParcelChat && selectedPolygon && (
        <ParcelChat
          parcel={selectedPolygon}
          nearestPark={nearestPark}
          nearestTransitStop={nearestTransitStop}
          onClose={() => setShowParcelChat(false)}
        />
      )}

      {/* Filter Modal */}
      {showFilterModal && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 2000,
          }}
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setShowFilterModal(false);
            }
          }}
        >
          <div
            style={{
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "24px",
              maxWidth: "600px",
              maxHeight: "80vh",
              overflowY: "auto",
              boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
              width: "90%",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "20px",
                borderBottom: "1px solid #e5e7eb",
                paddingBottom: "12px",
              }}
            >
              <h2 style={{ margin: 0, fontSize: "20px", fontWeight: "bold" }}>Filter Options</h2>
              <button
                onClick={() => setShowFilterModal(false)}
                style={{
                  border: "none",
                  background: "transparent",
                  cursor: "pointer",
                  fontSize: "24px",
                  color: "#6b7280",
                  padding: "0",
                  width: "30px",
                  height: "30px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                ✖
              </button>
            </div>

            {/* Land Type Filter */}
            {/*
            <div style={{ marginBottom: "24px" }}>
              <h3 style={{ marginBottom: "12px", fontSize: "16px", fontWeight: "600" }}>Land Type</h3>
              <div style={{ maxHeight: "150px", overflowY: "auto", border: "1px solid #e5e7eb", borderRadius: "4px", padding: "8px" }}>
                {filterOptions.landTypes.map((type) => (
                  <label
                    key={type}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      padding: "6px 0",
                      cursor: "pointer",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={filters.landTypes.includes(type)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setFilters({ ...filters, landTypes: [...filters.landTypes, type] });
                        } else {
                          setFilters({ ...filters, landTypes: filters.landTypes.filter((t) => t !== type) });
                        }
                      }}
                      style={{ marginRight: "8px", cursor: "pointer" }}
                    />
                    <span>{type || "N/A"}</span>
                  </label>
                ))}
              </div>
            </div>
            */}

            {/* Council District Filter */}
            <div style={{ marginBottom: "24px" }}>
              <h3 style={{ marginBottom: "12px", fontSize: "16px", fontWeight: "600" }}>Owner</h3>
              <div style={{ maxHeight: "150px", overflowY: "auto", border: "1px solid #e5e7eb", borderRadius: "4px", padding: "8px" }}>
                {filterOptions.owners.map((owner) => (
                  <label
                    key={owner}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      padding: "6px 0",
                      cursor: "pointer",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={filters.owners.includes(owner)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setFilters({ ...filters, owners: [...filters.owners, owner] });
                        } else {
                          setFilters({ ...filters, owners: filters.owners.filter((o) => o !== owner) });
                        }
                      }}
                      style={{ marginRight: "8px", cursor: "pointer" }}
                    />
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={owner}>{owner || "N/A"}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Council District Filter */}
            <div style={{ marginBottom: "24px" }}>
              <h3 style={{ marginBottom: "12px", fontSize: "16px", fontWeight: "600" }}>Council District</h3>
              <div style={{ maxHeight: "150px", overflowY: "auto", border: "1px solid #e5e7eb", borderRadius: "4px", padding: "8px" }}>
                {filterOptions.councilDistricts.map((district) => (
                  <label
                    key={district}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      padding: "6px 0",
                      cursor: "pointer",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={filters.councilDistricts.includes(district)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setFilters({ ...filters, councilDistricts: [...filters.councilDistricts, district] });
                        } else {
                          setFilters({ ...filters, councilDistricts: filters.councilDistricts.filter((d) => d !== district) });
                        }
                      }}
                      style={{ marginRight: "8px", cursor: "pointer" }}
                    />
                    <span>District {district}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Zoning District Filter */}
            <div style={{ marginBottom: "24px" }}>
              <h3 style={{ marginBottom: "12px", fontSize: "16px", fontWeight: "600" }}>Zoning District</h3>
              <div style={{ maxHeight: "150px", overflowY: "auto", border: "1px solid #e5e7eb", borderRadius: "4px", padding: "8px" }}>
                {filterOptions.zoningDistricts.map((zoning) => (
                  <label
                    key={zoning}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      padding: "6px 0",
                      cursor: "pointer",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={filters.zoningDistricts.includes(zoning)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setFilters({ ...filters, zoningDistricts: [...filters.zoningDistricts, zoning] });
                        } else {
                          setFilters({ ...filters, zoningDistricts: filters.zoningDistricts.filter((z) => z !== zoning) });
                        }
                      }}
                      style={{ marginRight: "8px", cursor: "pointer" }}
                    />
                    <span>{zoning || "N/A"}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* ZIP Code Filter */}
            <div style={{ marginBottom: "24px" }}>
              <h3 style={{ marginBottom: "12px", fontSize: "16px", fontWeight: "600" }}>ZIP Code</h3>
              <div style={{ maxHeight: "150px", overflowY: "auto", border: "1px solid #e5e7eb", borderRadius: "4px", padding: "8px" }}>
                {filterOptions.zipCodes.map((zip) => (
                  <label
                    key={zip}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      padding: "6px 0",
                      cursor: "pointer",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={filters.zipCodes.includes(zip)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setFilters({ ...filters, zipCodes: [...filters.zipCodes, zip] });
                        } else {
                          setFilters({ ...filters, zipCodes: filters.zipCodes.filter((z) => z !== zip) });
                        }
                      }}
                      style={{ marginRight: "8px", cursor: "pointer" }}
                    />
                    <span>{zip}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Land Rank Range Filter */}
            <div style={{ marginBottom: "24px" }}>
              <h3 style={{ marginBottom: "12px", fontSize: "16px", fontWeight: "600" }}>Land Rank Range</h3>
              <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                <div style={{ flex: 1 }}>
                  <label style={{ display: "block", marginBottom: "4px", fontSize: "14px", color: "#6b7280" }}>Min</label>
                  <input
                    type="number"
                    value={filters.minLandRank || ""}
                    onChange={(e) => setFilters({ ...filters, minLandRank: e.target.value ? Number(e.target.value) : null })}
                    placeholder="Min"
                    style={{
                      width: "100%",
                      padding: "8px",
                      border: "1px solid #e5e7eb",
                      borderRadius: "4px",
                      fontSize: "14px",
                    }}
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <label style={{ display: "block", marginBottom: "4px", fontSize: "14px", color: "#6b7280" }}>Max</label>
                  <input
                    type="number"
                    value={filters.maxLandRank || ""}
                    onChange={(e) => setFilters({ ...filters, maxLandRank: e.target.value ? Number(e.target.value) : null })}
                    placeholder="Max"
                    style={{
                      width: "100%",
                      padding: "8px",
                      border: "1px solid #e5e7eb",
                      borderRadius: "4px",
                      fontSize: "14px",
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Shape Area Range Filter */}
            <div style={{ marginBottom: "24px" }}>
              <h3 style={{ marginBottom: "12px", fontSize: "16px", fontWeight: "600" }}>Shape Area Range</h3>
              <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                <div style={{ flex: 1 }}>
                  <label style={{ display: "block", marginBottom: "4px", fontSize: "14px", color: "#6b7280" }}>Min</label>
                  <input
                    type="number"
                    value={filters.minShapeArea || ""}
                    onChange={(e) => setFilters({ ...filters, minShapeArea: e.target.value ? Number(e.target.value) : null })}
                    placeholder="Min"
                    style={{
                      width: "100%",
                      padding: "8px",
                      border: "1px solid #e5e7eb",
                      borderRadius: "4px",
                      fontSize: "14px",
                    }}
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <label style={{ display: "block", marginBottom: "4px", fontSize: "14px", color: "#6b7280" }}>Max</label>
                  <input
                    type="number"
                    value={filters.maxShapeArea || ""}
                    onChange={(e) => setFilters({ ...filters, maxShapeArea: e.target.value ? Number(e.target.value) : null })}
                    placeholder="Max"
                    style={{
                      width: "100%",
                      padding: "8px",
                      border: "1px solid #e5e7eb",
                      borderRadius: "4px",
                      fontSize: "14px",
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Precomputed Score Filters (0-10) */}
            <div style={{ marginBottom: "24px" }}>
              <h3 style={{ marginBottom: "12px", fontSize: "16px", fontWeight: "600" }}>Geographic Scores (0–10)</h3>

              {[
                ["Environmental", "minEnvironmentalScore", "maxEnvironmentalScore"],
                ["Recreational", "minRecreationalScore", "maxRecreationalScore"],
                ["Transit Accessibility", "minTransitScore", "maxTransitScore"],
                ["Walkability", "minWalkabilityScore", "maxWalkabilityScore"],
              ].map(([label, minKey, maxKey]) => (
                <div key={label} style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>{label}</div>
                  <div style={{ display: "flex", gap: 12 }}>
                    <input
                      type="number"
                      min={0}
                      max={10}
                      step={0.1}
                      value={filters[minKey] ?? ""}
                      onChange={(e) => setFilters({ ...filters, [minKey]: e.target.value ? Number(e.target.value) : null })}
                      placeholder="Min"
                      style={{
                        flex: 1,
                        padding: "8px",
                        border: "1px solid #e5e7eb",
                        borderRadius: "4px",
                        fontSize: "14px",
                      }}
                    />
                    <input
                      type="number"
                      min={0}
                      max={10}
                      step={0.1}
                      value={filters[maxKey] ?? ""}
                      onChange={(e) => setFilters({ ...filters, [maxKey]: e.target.value ? Number(e.target.value) : null })}
                      placeholder="Max"
                      style={{
                        flex: 1,
                        padding: "8px",
                        border: "1px solid #e5e7eb",
                        borderRadius: "4px",
                        fontSize: "14px",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Action Buttons */}
            <div style={{ display: "flex", gap: "12px", justifyContent: "flex-end", marginTop: "24px", borderTop: "1px solid #e5e7eb", paddingTop: "16px" }}>
              <button
                onClick={() => {
                  setFilters({
                    landTypes: [],
                    councilDistricts: [],
                    zoningDistricts: [],
                    zipCodes: [],
                    owners: [],
                    minLandRank: null,
                    maxLandRank: null,
                    minShapeArea: null,
                    maxShapeArea: null,
                    minEnvironmentalScore: null,
                    maxEnvironmentalScore: null,
                    minRecreationalScore: null,
                    maxRecreationalScore: null,
                    minTransitScore: null,
                    maxTransitScore: null,
                    minWalkabilityScore: null,
                    maxWalkabilityScore: null,
                  });
                }}
                style={{
                  padding: "10px 20px",
                  border: "1px solid #d1d5db",
                  borderRadius: "6px",
                  backgroundColor: "white",
                  cursor: "pointer",
                  fontSize: "14px",
                  fontWeight: "500",
                  color: "#374151",
                }}
              >
                Clear All
              </button>
              <button
                onClick={() => {
                  console.log("Applied filters:", filters);
                  setShowFilterModal(false);
                  // Filters are automatically applied via filteredMapData memo
                }}
                style={{
                  padding: "10px 20px",
                  border: "none",
                  borderRadius: "6px",
                  backgroundColor: "#3b82f6",
                  cursor: "pointer",
                  fontSize: "14px",
                  fontWeight: "500",
                  color: "white",
                }}
              >
                Apply Filters
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
