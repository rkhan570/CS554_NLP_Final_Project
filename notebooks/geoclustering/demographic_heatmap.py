import folium
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium

def load_and_merge_data():
    age_df = pd.read_csv('../../datasets/cleaned_datasets/demographic_dataset/age.csv')
    education_df = pd.read_csv('../../datasets/cleaned_datasets/demographic_dataset/education.csv')
    income_df = pd.read_csv('../../datasets/cleaned_datasets/demographic_dataset/income.csv')

    merged_df = age_df.merge(education_df, on='Area Code', how='inner')
    merged_df = merged_df.merge(income_df, on='Area Code', how='inner')

    merged_df.columns = merged_df.columns.str.strip().str.lower().str.replace(' ', '_')
    merged_df['area_code'] = merged_df['area_code'].astype(str)
    return merged_df

def load_geojson_data():
    geojson_path = "../../datasets/cleaned_datasets/philly_boundaries/Zipcodes_Poly.geojson"
    geojson = gpd.read_file(geojson_path)
    geojson['CODE'] = geojson['CODE'].str.strip()
    return geojson

def merge_demographics_with_geojson(merged_df, geojson):
    merged_df.rename(columns={'area_code': 'CODE'}, inplace=True)
    merged_df['CODE'] = merged_df['CODE'].str.strip()

    merged_gdf = geojson.merge(merged_df, on='CODE', how='left')
    return merged_gdf

def update_heatmap(m, selected_column, gdf):
    filtered_gdf = gdf[gdf[selected_column].notnull()]

    filtered_gdf[selected_column] = pd.to_numeric(filtered_gdf[selected_column], errors='coerce')

    heat_data = [
        [point['geometry'].centroid.y, point['geometry'].centroid.x, point[selected_column]] 
        for _, point in filtered_gdf.iterrows()
        if pd.notnull(point[selected_column])
    ]

    for layer in m._children:
        if isinstance(m._children[layer], folium.plugins.HeatMap):
            del m._children[layer]

    HeatMap(heat_data).add_to(m)
    return m

def create_streamlit_ui(merged_gdf, demographic_columns):
    st.title("Interactive Demographic Heatmap")
    st.write("Select a demographic column to visualize on the map:")

    selected_column = st.selectbox("Choose a demographic column", demographic_columns)

    m = folium.Map(location=[39.9526, -75.1652], zoom_start=12, 
                   scrollWheelZoom=False, zoomControl=False, dragging=False)

    folium.GeoJson(
        merged_gdf,
        name="ZIP Code Boundaries",
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.2
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['CODE', 'total_population', 'median_earnings_(dollars)'],
            aliases=["ZIP Code:", "Total Population:", "Median Earnings:"],
            localize=True
        ),
        popup=folium.GeoJsonPopup(fields=['CODE', 'total_population', 'median_earnings_(dollars)'])
    ).add_to(m)

    m = update_heatmap(m, selected_column, merged_gdf)

    bounds = merged_gdf.geometry.total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    st_folium(m, width=725)

def main():
    merged_df = load_and_merge_data()
    geojson = load_geojson_data()
    merged_gdf = merge_demographics_with_geojson(merged_df, geojson)

    demographic_columns = [
        'total_population', 'age', 'under_5_years', '5_to_9_years', '10_to_14_years', '15_to_19_years',
        '20_to_24_years', '25_to_29_years', '30_to_34_years', '35_to_39_years', '40_to_44_years', 
        '45_to_49_years', '50_to_54_years', '55_to_59_years', '60_to_64_years', '65_to_69_years', 
        '70_to_74_years', '75_to_79_years', '80_to_84_years', '85_years_and_over', 'less_than_high_school_graduate',
        'high_school_graduate', 'some_college_or_associate\'s_degree', 'bachelor\'s_degree_or_higher',
        'population_16_years_and_over_with_earnings', 'median_earnings_(dollars)', 'full-time,_year-round_workers_with_earnings',
        '$1_to_$9,999_or_loss', '$10,000_to_$14,999', '$15,000_to_$24,999', '$25,000_to_$34,999',
        '$35,000_to_$49,999', '$50,000_to_$64,999', '$65,000_to_$74,999', '$75,000_to_$99,999', '$100,000_or_more'
    ]

    create_streamlit_ui(merged_gdf, demographic_columns)

if __name__ == "__main__":
    main()
