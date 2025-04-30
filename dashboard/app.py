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

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Philadelphia Business Insights by Neighborhood")

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
    data = {}
    data['neighborhoods'] = gpd.read_file(
        cfg["boundaries_path"], engine="pyogrio"
    ).to_crs(cfg["crs_epsg"])
    base_path = cfg["precompute_dir"]
    with open(os.path.join(base_path, "heat_data.json")) as f:
        data['heat_data'] = json.load(f)
    biz_cols = ['name', 'stars_rev', 'sentiment_score', 'topic_label', 'review_count', 'geometry', 'cluster_id']
    data['biz'] = gpd.read_file(
        os.path.join(base_path, "businesses.geojson"),
        include_fields=biz_cols
    ).to_crs(cfg["crs_epsg"])
    data['clusters'] = gpd.read_file(
        os.path.join(base_path, "clusters.geojson"), engine="pyogrio"
    ).to_crs(cfg["crs_epsg"])
    return data

data = load_all_data()
neighborhoods, heat_data, biz, clusters = data.values()

# ==============================================
# FILTER CONTROLS
# ==============================================
with st.sidebar:
    st.header("Data Filters")
    demo_vars = sorted(heat_data.keys())
    selected_var = st.selectbox("Demographic Variable", demo_vars,
                                help="Select which demographic metric to visualize")
    col1, col2 = st.columns(2)
    show_heat = col1.checkbox("Heatmap", True)
    show_boundaries = col2.checkbox("Neighborhood Boundaries", True)

    st.subheader("Business Filters")
    show_biz = st.checkbox("Show Businesses", True)

    unique_neighborhoods = sorted(neighborhoods['CODE'].unique())
    selected_neighborhoods = st.multiselect(
        "Filter by Zip Code", unique_neighborhoods, default=[]
    )

    unique_topics = sorted(biz['topic_label'].unique())
    selected_topics = st.multiselect("Filter by Topic", unique_topics, default=[])

    sentiment_range = st.slider("Sentiment Range",
                                float(biz['sentiment_score'].min()),
                                float(biz['sentiment_score'].max()),
                                (-1.0, 1.0))

    star_range = st.slider("Star Rating Range",
                            float(biz['stars_rev'].min()),
                            float(biz['stars_rev'].max()),
                            (1.0, 5.0))

    st.subheader("Cluster Options")
    show_clust = st.checkbox("Show Clusters", True)
    # default to clustering by business count
    cluster_color_by = st.selectbox(
        "Color Clusters By",
        ["business_count", "avg_sentiment", "avg_stars"],
        index=0
    )
    min_cluster_size = st.slider("Minimum Cluster Size", 1, 100, 5)

# ==============================================
# DATA PROCESSING
# ==============================================
biz_filtered = biz[
    biz['sentiment_score'].between(*sentiment_range) &
    biz['stars_rev'].between(*star_range)
]

if selected_neighborhoods:
    selected_geoms = neighborhoods[
        neighborhoods['CODE'].isin(selected_neighborhoods)
    ].geometry
    biz_filtered = biz_filtered[
        biz_filtered.geometry.within(selected_geoms.unary_union)
    ]

if selected_topics:
    biz_filtered = biz_filtered[
        biz_filtered['topic_label'].isin(selected_topics)
    ]

clusters_filtered = clusters[
    clusters['business_count'] >= min_cluster_size
]

def prepare_geodata(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
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

biz_df = prepare_geodata(biz_filtered)
clusters_df = prepare_geodata(clusters_filtered)
neighborhoods_df = prepare_geodata(neighborhoods)

# ==============================================
# INTERACTIVE CLUSTER FUNCTIONS
# ==============================================
def get_businesses_in_cluster(cluster_id):
    return biz_filtered[biz_filtered['cluster_id'] == cluster_id]

def handle_cluster_click(info):
    if info.picked and info.object:
        props = info.object['properties']
        st.session_state['selected_cluster'] = {
            'id': props['cluster_id'],
            'businesses': get_businesses_in_cluster(props['cluster_id'])
        }

# ==============================================
# PYDECK VISUALIZATION
# ==============================================
view_state = pdk.ViewState(
    latitude=cfg["map_center"][0],
    longitude=cfg["map_center"][1],
    zoom=cfg["map_zoom"],
    pitch=0,
    bearing=0
)

layers = []

# Neighborhood boundaries (transparent fill + hoverable)
if show_boundaries:
    neighborhoods_geojson = json.loads(neighborhoods.to_json())
    boundary_layer = pdk.Layer(
        "GeoJsonLayer",
        data=neighborhoods_geojson,
        pickable=True,
        stroked=True,
        filled=True,                   # enable interior fill
        extruded=False,
        get_fill_color=[0, 0, 0, 0],   # fully transparent fill
        get_line_color=[0, 0, 0, 200],
        get_line_width=100,
        auto_highlight=True
    )
    layers.append(boundary_layer)

# Heatmap
if show_heat and selected_var in heat_data:
    heat_df = pd.DataFrame(heat_data[selected_var], columns=["lat", "lng", "weight"])
    layers.append(pdk.Layer(
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
    ))

# Business markers (pastel + stronger opacity + tooltip)
if show_biz and not biz_df.empty:
    biz_df["color"] = biz_df["sentiment_score"].apply(
        lambda x: [186, 255, 201, 200] if x >= 0 else [255, 179, 186, 200]
    )
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=biz_df,
        get_position="coordinates",
        get_color="color",
        get_radius=5,
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=8,
        pickable=True,
        tooltip={
            "html": """
                <b>{name}</b><br/>
                ⭐ {stars_rev:.1f} 💬 {review_count}<br/>
                😊 {sentiment_score:.2f}<br/>
                📂 {topic_label}
            """,
            "style": {
                "backgroundColor": "rgba(255,255,255,0.9)",
                "color": "#000",
                "fontSize": "12px",
                "padding": "5px"
            }
        }
    ))

# Cluster layer (unchanged)
if show_clust and not clusters_df.empty:
    metr = clusters_df[cluster_color_by]
    norm = (metr - metr.min()) / (metr.max() - metr.min())
    clusters_df["color"] = norm.apply(lambda x: [int(255*(1-x)), int(255*x), 0, 180])
    clusters_geojson = json.loads(clusters_filtered.to_json())
    layers.append(pdk.Layer(
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
        highlight_color=[255, 255, 0, 200],
        onClick=handle_cluster_click,
        tooltip={
            "html": """
                <b>Cluster #{cluster_id}</b><br/>
                Businesses: {business_count}<br/>
                Avg ⭐: {avg_stars:.1f}<br/>
                Avg 😊: {avg_sentiment:.2f}
            """,
            "style": {
                "backgroundColor": "rgba(255,255,255,0.9)",
                "color": "#000",
                "fontSize": "12px",
                "padding": "5px"
            }
        }
    ))

# Build the deck with a global tooltip for Zip Codes
deck = pdk.Deck(
    map_style=cfg["map_style"],
    initial_view_state=view_state,
    layers=layers,
    tooltip={
        "html": "<b>Zip Code:</b> {CODE}<br/>",
        "style": {
            "backgroundColor": "rgba(255,255,255,0.9)",
            "color": "#000",
            "fontSize": "12px",
            "padding": "5px"
        }
    }
)

st.pydeck_chart(deck)
# ==============================================
# CLUSTER ACTION DISPLAY
# ==============================================
if 'selected_cluster' in st.session_state:
    with st.container():
        st.subheader(f"Cluster #{st.session_state['selected_cluster']['id']}")
        cd = clusters.loc[
            clusters['cluster_id'] == st.session_state['selected_cluster']['id']
        ].iloc[0]
        col1, col2 = st.columns(2)
        col1.metric("Business Count", cd['business_count'])
        col1.metric("Avg Stars", f"{cd['avg_stars']:.1f}")
        col2.metric("Avg Sentiment", f"{cd['avg_sentiment']:.2f}")
        col2.metric("Main Topic", cd.get('primary_topic', 'N/A'))
        st.subheader(f"Businesses ({len(st.session_state['selected_cluster']['businesses'])})")
        df = st.session_state['selected_cluster']['businesses'][[
            'name', 'stars_rev', 'sentiment_score', 'topic_label'
        ]]
        df.rename(columns={
            'name': 'Name', 'stars_rev': 'Stars',
            'sentiment_score': 'Sentiment', 'topic_label': 'Topic'
        }, inplace=True)
        st.dataframe(df.style.format({
            'Stars': '{:.1f}', 'Sentiment': '{:.2f}'
        }), height=400, use_container_width=True)
        if st.button("Back to Map"):
            del st.session_state['selected_cluster']
            st.experimental_rerun()

# ==============================================
# ANALYTICS DASHBOARD
# ==============================================
st.header("Neighborhood Business Insights")
c1, c2, c3 = st.columns(3)
c1.metric("Visible Businesses", len(biz_filtered))
c2.metric("Avg Sentiment", f"{biz_filtered['sentiment_score'].mean():.2f}")
c3.metric("Avg Rating", f"{biz_filtered['stars_rev'].mean():.2f}")

tab1, tab2 = st.tabs(["Neighborhood Analysis", "Topic Analysis"])

with tab1:
    # spatial join
    biz_with_neighborhoods = gpd.sjoin(
        biz_filtered, neighborhoods, how="left", predicate="within"
    )
    if not biz_with_neighborhoods.empty:
        # 1) build your counts DataFrame
        counts = (
            biz_with_neighborhoods['CODE']
            .value_counts()
            .reset_index()
        )
        counts.columns = ['Zip Code', 'Business Count']
        
        # 2) convert ZIP to string
        counts['Zip Code'] = counts['Zip Code'].astype(str)
        
        # 3) make the bar chart, forcing x as categorical
        fig1 = px.bar(
            counts,
            x='Zip Code',
            y='Business Count',
            title="Business Count by Zip Code",
            category_orders={'Zip Code': sorted(counts['Zip Code'])}
        )
        fig1.update_xaxes(type='category')                                    # ← categorical axis
        fig1.update_traces(
            hovertemplate="Zip Code: %{x}<br>Business Count: %{y}<extra></extra>"
        )
        st.plotly_chart(fig1, use_container_width=True)

