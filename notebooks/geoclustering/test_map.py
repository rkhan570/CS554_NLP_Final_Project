import geopandas as gpd
import pandas as pd
import folium

# =============================================================================
# 1. LOAD THE SHAPEFILE AND DISSOLVE BY ZIP CODE
# =============================================================================
# Path to the TIGER/Line shapefile (make sure all related files are in the same folder)
shapefile_path = '../../datasets/cleaned_datasets/philly_boundaries/tl_2023_42101_faces.shp'

# Read the shapefile
gdf = gpd.read_file(shapefile_path)
print("Shapefile columns:", gdf.columns)

# Dissolve the face geometries by the ZIP code field 'ZCTA5CE20'
zcta_gdf = gdf.dissolve(by='ZCTA5CE20').reset_index()
print("Dissolved GeoDataFrame sample:")
print(zcta_gdf[['ZCTA5CE20']].head())

# =============================================================================
# 2. LOAD ACS DEMOGRAPHIC DATA AND EXTRACT ZIP CODE INFORMATION
# =============================================================================
# Example uses the DP04 ACS data CSV
dp04_path = r"C:\Users\alche\Downloads\ACSDP5Y2023.DP04_2025-04-15T143523\ACSDP5Y2023.DP04-Data.csv"
dp04_df = pd.read_csv(dp04_path)
print("DP04 CSV columns:", dp04_df.columns)

# Extract the ZIP code from the 'GEO_ID' field by taking the last 5 characters
if 'GEO_ID' in dp04_df.columns:
    dp04_df['ZIP'] = dp04_df['GEO_ID'].astype(str).str[-5:]
else:
    raise ValueError("No 'GEO_ID' column found in the ACS data.")
    
print("Sample of DP04 data with extracted ZIP codes:")
print(dp04_df[['GEO_ID', 'ZIP']].head())

# =============================================================================
# 3. MERGE THE ZIP BOUNDARIES WITH THE ACS DATA
# =============================================================================
# Merge using 'ZCTA5CE20' from the spatial data and 'ZIP' from the ACS data.
merged_gdf = zcta_gdf.merge(dp04_df, left_on='ZCTA5CE20', right_on='ZIP', how='left')
print("Merged GeoDataFrame sample:")
print(merged_gdf.head())

# (Optional) Ensure that the spatial data is in WGS84 for web mapping.
if merged_gdf.crs != "EPSG:4326":
    merged_gdf = merged_gdf.to_crs("EPSG:4326")

# =============================================================================
# 4. LOAD THE DP04 METADATA AND CREATE A LABEL MAPPING
# =============================================================================
# Path to the DP04 metadata CSV
metadata_path = r'C:\Users\alche\Downloads\ACSDP5Y2023.DP04_2025-04-15T143523\ACSDP5Y2023.DP04-Column-Metadata.csv'
metadata_df = pd.read_csv(metadata_path)
print("DP04 Metadata Columns:", metadata_df.columns)

# Create a dictionary mapping variable codes to descriptive labels.
# Adjust the column names below if your metadata uses different headers.
meta_dict = dict(zip(metadata_df['Column Name'], metadata_df['Label']))

# Determine which DP04 columns exist in both the metadata and the merged data.
# This list will be used to show demographic details in the popup.
demographic_vars = [col for col in merged_gdf.columns if col.startswith("DP04_") and col in meta_dict]
print("Demographic variables to display:", demographic_vars)

# =============================================================================
# 5. DEFINE POPUP & ON-EACH-FEATURE FUNCTIONS
# =============================================================================
# Build the HTML content for the popup using metadata descriptions
def popup_function(feature):
    props = feature['properties']
    zip_code = props.get('ZCTA5CE20', 'N/A')
    popup_html = f"<strong>ZIP Code:</strong> {zip_code}<br>"
    # Loop over each demographic variable and add its label and value.
    for var in demographic_vars:
        if var in props:
            label = meta_dict.get(var, var)
            value = props.get(var, 'No Data')
            popup_html += f"<strong>{label}:</strong> {value}<br>"
    return popup_html

# =============================================================================
# 6. CREATE THE INTERACTIVE MAP WITH FOLIUM
# =============================================================================
# Initialize a Folium map centered on Philadelphia.
m = folium.Map(location=[39.9526, -75.1652], zoom_start=11)

# Define a simple style for the ZIP code polygons.
def style_function(feature):
    return {
        'fillColor': 'blue',
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.2
    }

# Convert the merged GeoDataFrame to a GeoJSON-like dict.
geojson_data = merged_gdf.__geo_interface__

# Add the GeoJson layer using an inline lambda for on_each_feature.
folium.GeoJson(
    geojson_data,
    name="ZIP Boundaries",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(fields=['ZCTA5CE20'], aliases=["ZIP Code:"], localize=True),
    on_each_feature=lambda feature, layer: layer.bindPopup(popup_function(feature))
).add_to(m)

folium.LayerControl().add_to(m)

# =============================================================================
# 7. SAVE THE INTERACTIVE MAP
# =============================================================================
map_output_path = 'interactive_zip_demo_map.html'
m.save(map_output_path)
print(f"Interactive map saved at: {map_output_path}")