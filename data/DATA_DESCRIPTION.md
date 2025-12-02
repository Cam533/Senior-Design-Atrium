# Data Files Description

Overview of data files in the `/data` directory and their use for open space, parks, and recreational land analysis.

## Parks and Recreation Data

### PPR_Properties.csv

Contains all Philadelphia Parks & Recreation properties. Includes park names, addresses, zip codes, property classifications (PARK, PATHWAY, CONSERVATION), PPR use types (GREENWAY, RECREATION_SITE, NEIGHBORHOOD_PARK, SQUARE_PLAZA), acreage, council districts, and police districts. About 505 properties.

### PPR_Trails.csv

Trail information managed by Philadelphia Parks & Recreation. Fields include trail names, types (MAJOR, MINOR, CONNECTOR), surfaces (HARD, SOFT), trail systems (WEST_FAIRMOUNT, WISSAHICKON, EAST_FAIRMOUNT), access types (pedestrian, bike, horse), circuit trail designations, mileage, and status. About 1,888 trail segments.

### Existing_Trails.csv

Additional trail data with detailed specifications. Includes trail types (Designated), materials (Asphalt), width, paved status, and mileage. Trail names include Boxer's Trail, Centennial Loop, and others. About 340 trail segments.

### PPR_Program_Sites.csv

Recreation centers and program sites. Contains park and recreation center names, program types (PPR_REC), site classes, building presence, gym availability, and coordinates (X, Y). About 172 program sites.

### PPR_Districts_2018.csv

PPR administrative district boundaries. Includes district IDs (1-11), district labels, and shape areas/lengths for geographic boundaries. Used for organizing park data by administrative regions.

### ppr_tree_inventory_2024.csv

Tree inventory for Philadelphia parks. Contains tree species names (e.g., ACER PLATANOIDES - NORWAY MAPLE), diameter at breast height (DBH), year, and coordinates (loc_x, loc_y). About 151,714 trees.

### Registered_Community_Gardens.csv

Community gardens registered in Philadelphia. Includes garden names, associated park names, addresses, contact information, websites, hours of operation, garden status (ACTIVE), PPR land designation, and council districts. About 22 gardens.

## Land Use Data

### Land_Use (1).csv

City-wide land use classification for all parcels in Philadelphia. Primary fields are `c_dig1` (land use code) and `c_dig1desc` (description). Includes shape areas and lengths for geographic data. Year is 2023.

Land use codes:

- 1: Residential
- 2: Commercial
- 3: Industrial
- 4: Transportation
- 5: Cultural/Recreation
- 6: Utilities
- 7: Institutional
- 8: Open Space
- 9: Vacant
- 10: Water

For open space and recreation analysis, filter for codes 5 (Cultural/Recreation) and 8 (Open Space). About 559,738 parcels total.

## Transportation and Infrastructure Data

### Transit*Stops*(Spring_2025).csv

Public transit stops for Spring 2025. Contains stop coordinates (X, Y, Lat, Lon), stop names, line abbreviations, directions, and stop IDs. Includes bus, trolley, and other transit stops. About 22,479 stops.

### Bike_Network.csv

Bike network infrastructure. Contains bike lane segments with street names, lane types (Paint Buffered, Separated Bike Lane, Conventional), classes, and segment lengths. About 5,226 bike lane segments.

### CompleteStreets.csv

Complete streets data with comprehensive street infrastructure information. Includes sidewalk indicators (SIDEWALK_I), sidewalk width (SIDEWLK_WD), bike network facilities (BIKENETWOR), parking, street types, speed limits, and segment lengths. About 40,660 street segments.

### mini_city_halls.csv

Mini city halls and community centers. Contains facility names, addresses, community manager contacts, and coordinates (X, Y). About 13 locations.

## Other Data Files

### Vacant_Block_Percent_Building.csv

Vacant building statistics by block. Contains parcel counts, building vacancy counts and percentages. Not directly related to parks or recreation. About 22,723 blocks.

### philadelphia-pa-1.txt

Philadelphia Home Rule Charter text document. Legal document, not spatial or recreational data. About 152,719 lines.

### Development_Checklist-July-2024.pdf

Development checklist document. PDF format, contents not examined in detail.

### philadelphia_parcels_enriched.csv

Enriched parcel data with census tract information, geocoding, and demographic data. Used for contextual analysis of properties.

### Vacant_Indicators_Bldg.geojson

GeoJSON file containing vacant building indicators with geographic boundaries.

### Vacant_Indicators_Land.geojson

GeoJSON file containing vacant land indicators with geographic boundaries.

## Primary Data Sources for Analysis

For open space, parks, and recreational land use analysis, the main datasets are:

1. **Land_Use (1).csv** - Filter for codes 5 (Cultural/Recreation) and 8 (Open Space)
2. **PPR_Properties.csv** - All park properties
3. **PPR_Trails.csv** and **Existing_Trails.csv** - Trail networks
4. **PPR_Program_Sites.csv** - Recreation centers
5. **Registered_Community_Gardens.csv** - Community gardens
6. **ppr_tree_inventory_2024.csv** - Park tree inventory for environmental metrics
7. **Transit*Stops*(Spring_2025).csv** - Transit accessibility
8. **Bike_Network.csv** - Bike infrastructure
9. **CompleteStreets.csv** - Sidewalk and pedestrian infrastructure
10. **mini_city_halls.csv** - Community services

Supporting data:

- **PPR_Districts_2018.csv** - Administrative boundaries for organizing data
- **philadelphia_parcels_enriched.csv** - Contextual demographic data
