# -*- coding: utf-8 -*-
"""contextual recommendations.py

Analyzes land use data for Philadelphia, focusing on open space and recreational land uses.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the path to the data file relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.normpath(os.path.join(current_dir, "../../data/Land_Use (1).csv"))

# Load the land use data
print(f"Loading data from: {data_path}")
landuse = pd.read_csv(data_path)

# Display basic information about the dataset
print("\n=== Dataset Overview ===")
print(f"Total parcels: {len(landuse)}")
print("\nFirst few rows:")
print(landuse.head())

print("\nUnique land use categories:")
print(landuse['c_dig1desc'].unique())

print("\nLand use distribution (percentages):")
dist = landuse['c_dig1desc'].value_counts(normalize=True).round(3)*100
print(dist)

print("\nTop 10 land use combinations:")
print(landuse[['c_dig1desc', 'c_dig2desc']].value_counts().head(10))

code_labels = {
    1: "Residential",
    2: "Commercial",
    3: "Industrial",
    4: "Transportation",
    5: "Cultural/Recreation",
    6: "Utilities",
    7: "Institutional",
    8: "Open Space",
    9: "Vacant",
    10: "Water"
}

counts = landuse['c_dig1'].value_counts().sort_index()
labels = [code_labels.get(i, str(i)) for i in counts.index]


# Create visualization
print("\n=== Creating visualization ===")
plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color='skyblue', edgecolor='black')
plt.title("Distribution of Land Use Types in Philadelphia (2023)", fontsize=14)
plt.xlabel("Land Use Category", fontsize=12)
plt.ylabel("Number of Parcels", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot instead of showing it (for script execution)
output_dir = os.path.join(current_dir, "../../output")
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "land_use_distribution.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.close()  # Close the figure to free memory

print("\n=== Land Use Code Mapping ===")
print(landuse[['c_dig1', 'c_dig1desc']].drop_duplicates().sort_values('c_dig1'))

print("\n=== Area Analysis ===")
area_summary = landuse.groupby('c_dig1desc')['Shape__Area'].sum().sort_values(ascending=False)
area_percent = (area_summary / area_summary.sum() * 100).round(2)

print("Land Use by Total Area (sq meters):\n", area_summary)
print("\nLand Use by Percent of City Area:\n", area_percent)

# Focus on Open Space and Recreation
print("\n=== Open Space and Recreation Analysis ===")
open_space = landuse[landuse['c_dig1'] == 8]
recreation = landuse[landuse['c_dig1'] == 5]

print(f"Open Space parcels: {len(open_space)}")
open_space_area = open_space['Shape__Area'].sum()
recreation_area = recreation['Shape__Area'].sum()
print(f"Open Space total area: {open_space_area:,.2f} sq meters")
print(f"Recreation parcels: {len(recreation)}")
print(f"Recreation total area: {recreation_area:,.2f} sq meters")
combined_parcels = len(open_space) + len(recreation)
print(f"Combined Open Space + Recreation: {combined_parcels} parcels")
combined_area = open_space_area + recreation_area
print(f"Combined area: {combined_area:,.2f} sq meters")

#   - land use polygons: https://opendataphilly.org/datasets/land-use/
#   - ZIP code boundaries: https://opendataphilly.org/datasets/zip-codes/

# load datasets into GeoDataFrames
#   import geopandas as gpd
#   landuse = gpd.read_file("/content/Land_Use.shp").to_crs(epsg=4326)
#   zipcodes = gpd.read_file("/content/Zip_Codes.shp").to_crs(epsg=4326)

# spatially join to assign ZIP codes to parcels
#   landuse_zip = gpd.sjoin(landuse, zipcodes[["ZIPCODE","geometry"]], how="left", predicate="intersects")

# aggregate by ZIP code
#   summary = (
#       landuse_zip.groupby(["ZIPCODE","c_dig1desc"])["Shape__Area"]
#       .sum()
#       .reset_index()
#   )
#   summary.to_csv("landuse_by_zipcode.csv", index=False)
