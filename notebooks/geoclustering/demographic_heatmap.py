import warnings
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
from shapely.geometry import Point
from streamlit.components.v1 import html

st.set_page_config(
    page_title="Philly Demographics & Reviews",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_zip_demo():
    zip_gdf = (
        gpd.read_file("../../datasets/cleaned_datasets/philly_boundaries/Zipcodes_Poly.geojson")
        .to_crs("EPSG:4326")
    )
    zip_gdf['CODE'] = zip_gdf['CODE'].str.strip()
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
    pts = zip_gdf.geometry.representative_point()
    zip_gdf['centroid_x'] = pts.x
    zip_gdf['centroid_y'] = pts.y

    demo_base = "../../datasets/cleaned_datasets/demographic_dataset"
    df = (
        pd.read_csv(f"{demo_base}/age.csv")
        .merge(pd.read_csv(f"{demo_base}/education.csv"), on="Area Code")
        .merge(pd.read_csv(f"{demo_base}/income.csv"), on="Area Code")
    )
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['area_code'] = df['area_code'].astype(str).str.strip()
    df.rename(columns={'area_code':'CODE'}, inplace=True)

    merged = zip_gdf.merge(df, on='CODE', how='left')
    return merged

@st.cache_data
def precompute_heat():
    gdf = load_zip_demo()
    exclude = {
        'CODE','geometry','centroid_x','centroid_y',
        'OBJECTID','COD','Shape__Area','Shape__Length',
        'unnamed:_0_x','age','unnamed:_0_y','unnamed:_0'
    }
    heat = {}
    for col in gdf.columns:
        if col in exclude: continue
        vals = pd.to_numeric(gdf[col], errors='coerce')
        if not vals.notna().any(): continue
        heat[col] = [
            [float(y), float(x), float(v)]
            for x,y,v in zip(gdf.centroid_x, gdf.centroid_y, vals)
            if pd.notna(v)
        ]
    return heat

@st.cache_data
def load_businesses():
    biz = gpd.read_file("gdf_businesses.geojson").to_crs("EPSG:4326")
    agg = (
        biz.groupby("business_id")
        .agg(
            latitude=("latitude","first"),
            longitude=("longitude","first"),
            sentiment_score=("sentiment_score","mean"),
            subjectivity_score=("subjectivity_score","mean"),
            stars_rev=("stars_rev","mean"),
            topic_label=("topic_label", lambda s: s.mode().iloc[0]),
            hdbscan_cluster=("hdbscan_cluster","first")
        )
        .reset_index()
    )
    return agg

@st.cache_data
def load_clusters():
    clusters = gpd.read_file("clusters.geojson").to_crs("EPSG:4326")
    datetime_cols = clusters.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    clusters.drop(columns=datetime_cols, inplace=True)
    clusters['popup_html'] = clusters.apply(
        lambda r: (
            f"<strong>Cluster {int(r.hdbscan_cluster)}</strong><br>"
            f"Count: {int(r.business_count)}<br>"
            f"Avg Stars: {r.avg_stars:.2f}<br>"
            f"Avg Sentiment: {r.avg_sentiment:.2f}<br>"
            f"Avg Subjectivity: {r.avg_subjectivity:.2f}"
        ), axis=1
    )
    return clusters

heat_data = precompute_heat()
demo_cols = list(heat_data.keys())

st.title("📊 Philadelphia Demographics & Reviews")
selected = st.selectbox("Choose demographic variable for heatmap:", demo_cols)

gdf = load_zip_demo()
m = folium.Map(
    location=[39.9526, -75.1652],
    zoom_start=12,
    tiles="CartoDB Positron",
    prefer_canvas=True
)

folium.GeoJson(
    gdf,
    name="ZIP Boundaries",
    smooth_factor=1,
    style_function=lambda _: {
        'fillColor':'blue','color':'black','weight':1,'fillOpacity':0.1
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['CODE', selected],
        aliases=['ZIP Code:', f"{selected.replace('_',' ').title()}:"])
).add_to(m)

HeatMap(
    heat_data[selected],
    name="Demographic Heatmap",
    radius=15, blur=25, max_zoom=12, min_opacity=0.5
).add_to(m)

biz_df = load_businesses()
marker_cluster = MarkerCluster(name="Businesses").add_to(m)
for _, r in biz_df.iterrows():
    folium.Marker(
        location=[r.latitude, r.longitude],
        popup=folium.Popup(html=(
            f"<strong>Stars:</strong> {r.stars_rev:.1f}<br>"
            f"<strong>Sentiment:</strong> {r.sentiment_score:.2f}<br>"
            f"<strong>Subjectivity:</strong> {r.subjectivity_score:.2f}<br>"
            f"<strong>Topic:</strong> {r.topic_label}"
        ), max_width=250),
        icon=folium.Icon(color='green' if r.sentiment_score>=0 else 'red')
    ).add_to(marker_cluster)

clusters = load_clusters()
folium.GeoJson(
    data=clusters.__geo_interface__,
    name="Clusters",
    smooth_factor=1,
    style_function=lambda _: {
        'fillColor':'orange','color':'darkorange','weight':2,'fillOpacity':0.2
    },
    popup=folium.GeoJsonPopup(fields=['popup_html'])
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
minx, miny, maxx, maxy = gdf.geometry.total_bounds
m.fit_bounds([[miny, minx], [maxy, maxx]])

map_html = m.get_root().render()
wrapped = f'<div style="width:100vw;height:80vh;margin:0;padding:0;">{map_html}</div>'
html(wrapped, height=800)
