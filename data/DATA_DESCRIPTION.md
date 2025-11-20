# Data Files Description

This document describes each data file in the `/data` directory and their relevance to open space, parks, and recreational land uses.

## Parks and Recreation Data (PPR = Philadelphia Parks & Recreation)

### 1. **PPR_Properties.csv** ✅
- **Purpose**: Contains all Philadelphia Parks & Recreation properties
- **Key Fields**: 
  - Park names, addresses, zip codes
  - Property classifications (PARK, PATHWAY, etc.)
  - PPR use types (GREENWAY, RECREATION_SITE, NEIGHBORHOOD_PARK, SQUARE_PLAZA)
  - Acreage, council districts, police districts
- **Relevance**: **Directly relevant** - This is the main dataset for parks and recreational properties
- **Records**: ~505 properties

### 2. **PPR_Trails.csv** ✅
- **Purpose**: Contains trail information managed by Philadelphia Parks & Recreation
- **Key Fields**:
  - Trail names, types (MAJOR), surfaces (HARD)
  - Trail systems (WEST_FAIRMOUNT, WISSAHICKON, EAST_FAIRMOUNT)
  - Access types (pedestrian, bike, horse)
  - Circuit trail designations (Schuylkill River Trail, etc.)
  - Miles, status (EXISTING)
- **Relevance**: **Directly relevant** - Trails are recreational open space
- **Records**: ~1,888 trails

### 3. **Existing_Trails.csv** ✅
- **Purpose**: Contains existing trail data with detailed specifications
- **Key Fields**:
  - Trail types (Designated), materials (Asphalt)
  - Width, paved status, mileage
  - Trail names (e.g., Boxer's Trail)
- **Relevance**: **Directly relevant** - Additional trail data for recreational use
- **Records**: ~340 trails

### 4. **PPR_Program_Sites.csv** ✅
- **Purpose**: Contains recreation centers and program sites
- **Key Fields**:
  - Park/recreation center names
  - Program types (PPR_REC)
  - Site classes, building presence, gym availability
  - Coordinates (X, Y)
- **Relevance**: **Directly relevant** - Recreation centers are part of recreational infrastructure
- **Records**: ~172 program sites

### 5. **PPR_Districts_2018.csv** ✅
- **Purpose**: Contains PPR administrative district boundaries
- **Key Fields**:
  - District IDs (1, 2, 3, 4)
  - District labels
  - Shape areas and lengths (geographic boundaries)
- **Relevance**: **Relevant** - Administrative boundaries for organizing park data
- **Records**: 11 districts

### 6. **ppr_tree_inventory_2024.csv** ✅
- **Purpose**: Tree inventory in Philadelphia parks
- **Key Fields**:
  - Tree species names (e.g., ACER PLATANOIDES - NORWAY MAPLE)
  - Tree diameter at breast height (DBH)
  - Year, coordinates (loc_x, loc_y)
- **Relevance**: **Relevant** - Trees are part of open space and park infrastructure
- **Records**: ~151,714 trees

### 7. **Registered_Community_Gardens.csv** ✅
- **Purpose**: Community gardens registered in Philadelphia
- **Key Fields**:
  - Garden names, park names, addresses
  - Contact information, websites
  - Hours of operation, garden status (ACTIVE)
  - PPR land designation, council districts
- **Relevance**: **Directly relevant** - Community gardens are recreational/open space uses
- **Records**: ~22 gardens

## Land Use Data

### 8. **Land_Use (1).csv** ✅
- **Purpose**: City-wide land use classification for all parcels in Philadelphia
- **Key Fields**:
  - Land use codes: `c_dig1` (primary) and `c_dig1desc` (description)
  - **Code 5 = "Cultural/Recreation"** - Recreational land uses
  - **Code 8 = "Open Space"** - Open space land uses
  - Shape areas and lengths (geographic data)
  - Year (2023)
- **Relevance**: **Directly relevant** - Contains open space (code 8) and cultural/recreation (code 5) categories
- **Records**: ~559,738 parcels
- **Note**: Based on `contextual_recommendations.py`, the land use codes are:
  - 1: Residential
  - 2: Commercial
  - 3: Industrial
  - 4: Transportation
  - **5: Cultural/Recreation** ⭐
  - 6: Utilities
  - 7: Institutional
  - **8: Open Space** ⭐
  - 9: Vacant
  - 10: Water

## Other Data

### 9. **Vacant_Block_Percent_Building.csv**
- **Purpose**: Vacant building statistics by block
- **Key Fields**: Parcel counts, building vacancy counts and percentages
- **Relevance**: **Not directly relevant** - This is about vacant buildings, not parks/recreation
- **Records**: ~22,723 blocks

### 10. **philadelphia-pa-1.txt**
- **Purpose**: Philadelphia Home Rule Charter text document
- **Relevance**: **Not relevant** - Legal document, not spatial/recreational data
- **Size**: ~152,719 lines

### 11. **Development_Checklist-July-2024.pdf**
- **Purpose**: Development checklist document
- **Relevance**: **Unknown** - PDF file, would need to examine contents

---

## Summary

**✅ Directly Relevant to Open Space, Parks, and Recreational Land Uses:**
1. PPR_Properties.csv - All PPR properties
2. PPR_Trails.csv - PPR trails
3. Existing_Trails.csv - Additional trail data
4. PPR_Program_Sites.csv - Recreation centers
5. Registered_Community_Gardens.csv - Community gardens
6. Land_Use (1).csv - Contains codes 5 (Cultural/Recreation) and 8 (Open Space)

**✅ Relevant (Supporting Data):**
7. PPR_Districts_2018.csv - Administrative boundaries
8. ppr_tree_inventory_2024.csv - Park trees

**❌ Not Relevant:**
9. Vacant_Block_Percent_Building.csv - Vacant building data
10. philadelphia-pa-1.txt - Legal charter document
11. Development_Checklist-July-2024.pdf - Unknown content

## Recommendation

For analyzing open space, parks, and recreational land uses, focus on:
- **Land_Use (1).csv** filtered for codes 5 and 8
- **PPR_Properties.csv** for all park properties
- **PPR_Trails.csv** and **Existing_Trails.csv** for trail networks
- **PPR_Program_Sites.csv** for recreation centers
- **Registered_Community_Gardens.csv** for community gardens

