"""
Philadelphia Yelp Dashboard with Interactive Cluster Hover Actions
"""
import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import geopandas as gpd
import pydeck as pdk
from typing import Dict, List

# Configuration
@st.cache_data
def get_config():
    return {
        "precompute_dir": "precomputed",
        "boundaries_path": "datasets/cleaned_datasets/philly_boundaries/Zipcodes_Poly.geojson",
        "crs_epsg": "EPSG:4326",
        "map_center": [39.9526, -75.1652],
        "map_zoom": 11,
        "map_style": "light"
    }

cfg = get_config()

# Cache all data in one function
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data():
    """Load and preprocess all data at once"""
    data = {}
    
    # Load neighborhood boundaries
    data['neighborhoods'] = gpd.read_file(
        cfg["boundaries_path"],
        engine="pyogrio"
    ).to_crs(cfg["crs_epsg"])
    
    # Load other data
    base_path = cfg["precompute_dir"]
    with open(os.path.join(base_path, "heat_data.json")) as f:
        data['heat_data'] = json.load(f)
    
    biz_cols = ['name', 'stars_rev', 'sentiment_score', 'topic_label', 'review_count', 'geometry', 'cluster_id']
    data['biz'] = gpd.read_file(
        os.path.join(base_path, "businesses.geojson"),
        include_fields=biz_cols
    ).to_crs(cfg["crs_epsg"])
    
    data['clusters'] = gpd.read_file(
        os.path.join(base_path, "clusters.geojson"),
        engine="pyogrio"
    ).to_crs(cfg["crs_epsg"])
    
    return data

# Load data
data = load_all_data()
neighborhoods, heat_data, biz, clusters = data.values()

# Initialize app
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Philadelphia Business Insights by Neighborhood")

# ==============================================
# FILTER CONTROLS
# ==============================================

with st.sidebar:
    st.header("Data Filters")
    
    # Demographic selection
    demo_vars = sorted(heat_data.keys())
    selected_var = st.selectbox(
        "Demographic Variable", 
        demo_vars,
        help="Select which demographic metric to visualize"
    )
    
    # Visualization toggles
    col1, col2 = st.columns(2)
    show_heat = col1.checkbox("Heatmap", True)
    show_boundaries = col2.checkbox("Neighborhood Boundaries", True)
    
    # Business filters
    st.subheader("Business Filters")
    show_biz = st.checkbox("Show Businesses", True)
    
    # Neighborhood selection
    unique_neighborhoods = sorted(neighborhoods['CODE'].unique())
    selected_neighborhoods = st.multiselect(
        "Filter by Zip Code",
        unique_neighborhoods,
        default=[]
    )
    
    # Topic selection
    unique_topics = sorted(biz['topic_label'].unique())
    selected_topics = st.multiselect(
        "Filter by Topic",
        unique_topics,
        default=[]
    )
    
    # Sentiment filter
    sentiment_range = st.slider(
        "Sentiment Range",
        float(biz['sentiment_score'].min()),
        float(biz['sentiment_score'].max()),
        (-1.0, 1.0)
    )
    
    # Star rating filter
    star_range = st.slider(
        "Star Rating Range",
        float(biz['stars_rev'].min()),
        float(biz['stars_rev'].max()),
        (1.0, 5.0)
    )
    
    # Cluster filters
    st.subheader("Cluster Options")
    show_clust = st.checkbox("Show Clusters", True)
    cluster_color_by = st.selectbox(
        "Color Clusters By",
        ["business_count", "avg_sentiment", "avg_stars"],
        index=1
    )
    min_cluster_size = st.slider(
        "Minimum Cluster Size",
        1, 100, 5
    )

# ==============================================
# DATA PROCESSING
# ==============================================

# Filter businesses
biz_filtered = biz[
    (biz['sentiment_score'].between(*sentiment_range)) &
    (biz['stars_rev'].between(*star_range))
]

# Filter by neighborhood if selected
if selected_neighborhoods:
    selected_geoms = neighborhoods[neighborhoods['CODE'].isin(selected_neighborhoods)].geometry
    biz_filtered = biz_filtered[biz_filtered.geometry.within(selected_geoms.unary_union)]

if selected_topics:
    biz_filtered = biz_filtered[biz_filtered['topic_label'].isin(selected_topics)]

# Filter clusters
clusters_filtered = clusters[clusters['business_count'] >= min_cluster_size]

def prepare_geodata(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Convert GeoDataFrame to PyDeck-ready DataFrame"""
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    
    if gdf.geom_type.iloc[0] == 'Point':
        df['coordinates'] = gdf.geometry.apply(
            lambda geom: [geom.x, geom.y] if not geom.is_empty else None
        )
    else:
        df['coordinates'] = gdf.geometry.apply(
            lambda geom: list(geom.centroid.coords)[0] if not geom.is_empty else None
        )
        df['geometry'] = gdf.geometry.apply(
            lambda geom: geom.__geo_interface__ if not geom.is_empty else None
        )
    
    return df.dropna(subset=['coordinates'])

# Prepare data for PyDeck
biz_df = prepare_geodata(biz_filtered)
clusters_df = prepare_geodata(clusters_filtered)
neighborhoods_df = prepare_geodata(neighborhoods)

# ==============================================
# INTERACTIVE CLUSTER FUNCTIONS
# ==============================================

def get_businesses_in_cluster(cluster_id):
    """Get businesses belonging to a specific cluster"""
    return biz_filtered[biz_filtered['cluster_id'] == cluster_id]

def handle_cluster_click(info):
    """Callback function for cluster click events"""
    if info.picked and info.object:
        cluster_data = info.object
        st.session_state['selected_cluster'] = {
            'id': cluster_data['properties']['cluster_id'],
            'businesses': get_businesses_in_cluster(cluster_data['properties']['cluster_id'])
        }

# ==============================================
# PYDECK VISUALIZATION
# ==============================================

# Create container for cluster actions
cluster_action_container = st.empty()

view_state = pdk.ViewState(
    latitude=cfg["map_center"][0],
    longitude=cfg["map_center"][1],
    zoom=cfg["map_zoom"],
    pitch=0,
    bearing=0
)

layers = []

# Neighborhood boundaries
if show_boundaries and not neighborhoods_df.empty:
    boundary_layer = pdk.Layer(
        "PolygonLayer",
        data=neighborhoods_df,
        get_polygon="geometry.coordinates",
        stroked=True,
        filled=False,
        extruded=False,
        get_line_color=[0, 0, 0, 200],
        get_line_width=100,
        pickable=True,
        auto_highlight=True,
    )
    layers.append(boundary_layer)

# Heatmap
if show_heat and selected_var in heat_data:
    heat_df = pd.DataFrame(heat_data[selected_var], columns=["lat", "lng", "weight"])
    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=heat_df,
        get_position=["lng", "lat"],
        get_weight="weight",
        opacity=0.8,
        threshold=0.5,
        color_range=[
            [0, 0, 255, 50],
            [0, 255, 255, 100],
            [0, 255, 0, 150],
            [255, 255, 0, 200],
            [255, 0, 0, 250]
        ],
        radius_pixels=30,
    )
    layers.append(heat_layer)

# Business markers
if show_biz and not biz_df.empty:
    biz_df["color"] = biz_df["sentiment_score"].apply(
        lambda x: [0, 255, 0, 200] if x >= 0 else [255, 0, 0, 200]
    )
    
    biz_layer = pdk.Layer(
        "ScatterplotLayer",
        data=biz_df,
        get_position="coordinates",
        get_color="color",
        get_radius=50,
        radius_scale=10,
        pickable=True,
    )
    layers.append(biz_layer)

# Interactive Cluster Layer
if show_clust and not clusters_df.empty:
    # Normalize cluster metric for coloring
    metric = clusters_df[cluster_color_by]
    normalized = (metric - metric.min()) / (metric.max() - metric.min())
    clusters_df["color"] = normalized.apply(
        lambda x: [int(255 * (1 - x)), int(255 * x), 0, 180]
    )
    
    # Convert to GeoJSON for better interactivity
    clusters_geojson = json.loads(clusters_filtered.to_json())
    
    cluster_layer = pdk.Layer(
        "GeoJsonLayer",
        data=clusters_geojson,
        opacity=0.6,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=False,
        get_fill_color="color",
        get_line_color=[0, 0, 0, 200],
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 0, 200],  # Yellow highlight on hover
        onClick=handle_cluster_click,
        tooltip={
            "html": """
                <b>Cluster #{cluster_id}</b><br/>
                <b>Businesses:</b> {business_count}<br/>
                <b>Avg Stars:</b> {avg_stars:.1f}<br/>
                <b>Avg Sentiment:</b> {avg_sentiment:.2f}
            """,
            "style": {
                "backgroundColor": "white",
                "color": "black",
                "fontSize": "14px"
            }
        }
    )
    layers.append(cluster_layer)

# Tooltip for neighborhoods
neighborhood_tooltip = {
    "html": "<b>Zip Code:</b> {CODE}<br/>",
    "style": {
        "backgroundColor": "white",
        "color": "black"
    }
}

# Render the map
deck = pdk.Deck(
    map_style=cfg["map_style"],
    initial_view_state=view_state,
    layers=layers,
    tooltip=neighborhood_tooltip
)

st.pydeck_chart(deck)

# ==============================================
# CLUSTER ACTION DISPLAY
# ==============================================

if 'selected_cluster' in st.session_state:
    with cluster_action_container.container():
        st.subheader(f"Cluster #{st.session_state['selected_cluster']['id']}")
        
        cluster_data = clusters[clusters['cluster_id'] == st.session_state['selected_cluster']['id']].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Business Count", cluster_data['business_count'])
            st.metric("Avg Stars", f"{cluster_data['avg_stars']:.1f}")
        
        with col2:
            st.metric("Avg Sentiment", f"{cluster_data['avg_sentiment']:.2f}")
            st.metric("Main Topic", cluster_data.get('primary_topic', 'N/A'))
        
        st.subheader(f"Businesses ({len(st.session_state['selected_cluster']['businesses'])})")
        biz_df = pd.DataFrame({
            'Name': st.session_state['selected_cluster']['businesses']['name'],
            'Stars': st.session_state['selected_cluster']['businesses']['stars_rev'],
            'Sentiment': st.session_state['selected_cluster']['businesses']['sentiment_score'],
            'Topic': st.session_state['selected_cluster']['businesses']['topic_label']
        })
        
        st.dataframe(
            biz_df.style.format({
                'Stars': '{:.1f}',
                'Sentiment': '{:.2f}'
            }),
            height=400,
            use_container_width=True
        )
        
        if st.button("Back to Map"):
            del st.session_state['selected_cluster']
            st.experimental_rerun()

# ==============================================
# ANALYTICS DASHBOARD
# ==============================================

st.header("Neighborhood Business Insights")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Visible Businesses", len(biz_filtered))
col2.metric("Avg Sentiment", f"{biz_filtered['sentiment_score'].mean():.2f}")
col3.metric("Avg Rating", f"{biz_filtered['stars_rev'].mean():.2f}")

# Charts
tab1, tab2 = st.tabs(["Neighborhood Analysis", "Topic Analysis"])

with tab1:
    biz_with_neighborhoods = gpd.sjoin(
        biz_filtered, 
        neighborhoods, 
        how="left", 
        predicate="within"
    )
    
    if not biz_with_neighborhoods.empty:
        neighborhood_counts = biz_with_neighborhoods['CODE'].value_counts().reset_index()
        neighborhood_counts.columns = ['Zip Code', 'Business Count']
        
        fig = px.bar(
            neighborhood_counts,
            x='Zip Code',
            y='Business Count',
            title="Business Count by Zip Code"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        neighborhood_sentiment = biz_with_neighborhoods.groupby('CODE')['sentiment_score'].mean().reset_index()
        
        fig = px.bar(
            neighborhood_sentiment,
            x='CODE',
            y='sentiment_score',
            title="Average Sentiment by Zip Code"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.pie(
        biz_filtered,
        names="topic_label",
        title="Business Distribution by Topic"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.box(
        biz_filtered,
        x="topic_label",
        y="stars_rev",
        title="Star Ratings by Topic"
    )
    st.plotly_chart(fig, use_container_width=True)