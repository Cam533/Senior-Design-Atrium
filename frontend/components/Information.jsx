import React from 'react'

export default function Information({ plotInfo }) {
  if (!plotInfo) {
    return (
      <div style={{ padding: '20px', color: '#666' }}>
        <p>No plot selected. Click on a vacant lot to view its information.</p>
      </div>
    )
  }

  // Helper function to format property names
  const formatKey = (key) => {
    return key
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase())
      .trim()
  }

  // Helper function to format values
  const formatValue = (value) => {
    if (value === null || value === undefined) return 'N/A'
    if (typeof value === 'number') {
      // Format large numbers with commas
      return value.toLocaleString()
    }
    return String(value)
  }

  // Priority fields to show at the top (if they exist)
  const priorityFields = ['address', 'zoning', 'objectid', 'zoningbasedistrict']
  
  // Get priority fields first, then all others
  const sortedKeys = [
    ...priorityFields.filter(key => plotInfo.hasOwnProperty(key)),
    ...Object.keys(plotInfo).filter(key => !priorityFields.includes(key))
  ]

  return (
    <div style={{ 
      padding: '20px', 
      height: '100%',
      overflowY: 'auto',
      backgroundColor: '#f8f9fa'
    }}>
      <h2 style={{ 
        marginTop: 0, 
        marginBottom: '20px',
        fontSize: '20px',
        color: '#1a1a1a',
        borderBottom: '2px solid #2b8cbe',
        paddingBottom: '10px'
      }}>
        Plot Information
      </h2>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        {sortedKeys.map((key) => {
          // Skip geometry-related fields (they're not useful to display as text)
          if (key === 'geometry' || key.toLowerCase().includes('geom')) {
            return null
          }

          const value = plotInfo[key]
          const isPriority = priorityFields.includes(key)

          return (
            <div 
              key={key}
              style={{
                padding: '12px',
                backgroundColor: 'white',
                borderRadius: '6px',
                border: isPriority ? '2px solid #2b8cbe' : '1px solid #e0e0e0',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
              }}
            >
              <div style={{
                fontSize: '12px',
                fontWeight: '600',
                color: '#666',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                marginBottom: '4px'
              }}>
                {formatKey(key)}
              </div>
              <div style={{
                fontSize: '15px',
                color: '#1a1a1a',
                wordBreak: 'break-word'
              }}>
                {formatValue(value)}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
