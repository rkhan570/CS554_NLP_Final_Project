"""
Streamlit dashboard for Philly Yelp + Demographics.
Loads precomputed GeoJSONs and heat data for fast startup.
"""

import os
import json
import yaml

import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit.components.v1 import html

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PRE_DIR   = cfg["precompute_dir"]
CRS_EPSG  = cfg["crs_epsg"]
MAP_CENTER= cfg.get("map_center", [39.9526, -75.1652])
MAP_ZOOM  = cfg.get("map_zoom", 12)

@st.cache_data
def load_zip_demo():
    path = os.path.join(PRE_DIR, "zip_demo.geojson")
    return gpd.read_file(path).to_crs(CRS_EPSG)

@st.cache_data
def load_heat_data():
    path = os.path.join(PRE_DIR, "heat_data.json")
    with open(path, "r") as fp:
        return json.load(fp)

@st.cache_data
def load_businesses():
    path = os.path.join(PRE_DIR, "businesses.geojson")
    return gpd.read_file(path).to_crs(CRS_EPSG)

@st.cache_data
def load_clusters():
    path = os.path.join(PRE_DIR, "clusters.geojson")
    return gpd.read_file(path).to_crs(CRS_EPSG)

st.set_page_config(
    page_title="Philly Yelp & Demographics",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Philadelphia Yelp & Demographics Dashboard")

zip_demo  = load_zip_demo()
heat_data = load_heat_data()
biz       = load_businesses()
clusters  = load_clusters()

demo_vars    = sorted(heat_data.keys())
selected_var = st.sidebar.selectbox("Select demographic variable for heatmap:", demo_vars)
show_biz     = st.sidebar.checkbox("Show business markers", value=True)
show_clust   = st.sidebar.checkbox("Show cluster polygons", value=True)

m = folium.Map(
    location=MAP_CENTER,
    zoom_start=MAP_ZOOM,
    tiles="CartoDB Positron"
)

folium.Choropleth(
    geo_data=zip_demo.__geo_interface__,
    data=zip_demo,
    columns=["CODE", selected_var],
    key_on="feature.properties.CODE",
    fill_opacity=0.3,
    line_opacity=0.2,
    legend_name=selected_var.replace("_", " ").title()
).add_to(m)

HeatMap(
    heat_data[selected_var],
    name="Heatmap",
    radius=15,
    blur=25,
    max_zoom=12,
    min_opacity=0.5
).add_to(m)

if show_biz:
    mc = MarkerCluster(name="Businesses").add_to(m)
    for _, r in biz.iterrows():
        folium.Marker(
            location=[r.geometry.y, r.geometry.x],
            popup=folium.Popup(
                html=f"""
                <strong>Stars:</strong> {r.stars_rev:.1f}<br>
                <strong>Sentiment:</strong> {r.sentiment_score:.2f}<br>
                <strong>Topic:</strong> {r.topic_label}
                """,
                max_width=250
            ),
            icon=folium.Icon(color="green" if r.sentiment_score >= 0 else "red")
        ).add_to(mc)

if show_clust:
    folium.GeoJson(
        data=clusters.__geo_interface__,
        name="Clusters",
        style_function=lambda feat: {
            "fillColor": "orange",
            "color": "darkorange",
            "weight": 2,
            "fillOpacity": 0.2
        },
        highlight_function=lambda feat: {
            "fillColor": "yellow",
            "color": "black",
            "weight": 3,
            "fillOpacity": 0.5
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["cluster_id", "business_count", "avg_sentiment"],
            aliases=["Cluster:", "Count:", "Avg Sentiment:"],
            localize=True
        ),
        popup=folium.GeoJsonPopup(
            fields=["cluster_id", "business_count", "avg_sentiment", "avg_subjectivity", "avg_stars"],
            aliases=["Cluster ID:", "Business Count:", "Avg Sentiment:", "Avg Subjectivity:", "Avg Stars:"],
            localize=True,
            labels=True
        )
    ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
minx, miny, maxx, maxy = zip_demo.total_bounds
m.fit_bounds([[miny, minx], [maxy, maxx]])

map_html = m.get_root().render()
html(f'<div style="width:100vw;height:80vh;">{map_html}</div>', height=800)
