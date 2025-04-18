"""
Batch‐process Yelp + demographic data into GeoJSONs for quick loading in Streamlit.

Loads settings from `config.yaml` in the same directory.

Outputs (to `precompute_dir`):
  - businesses.geojson  
  - clusters.geojson    
  - zip_demo.geojson    
  - heat_data.json      
"""
import os
import logging
import json
import warnings

import yaml
import nltk
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import hdbscan
import dotenv
from openai import OpenAI

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Load config
cfg = load_config()

INPUT_PARQUET       = cfg["reviews_parquet"]
OUTPUT_DIR          = cfg["precompute_dir"]
ZIP_BOUNDARIES_PATH = cfg["zip_boundaries"]
DEMOG_BASE_DIR      = cfg["demog_base_dir"]
NUM_TOPICS          = cfg["num_topics"]
MIN_CLUSTER_SIZE    = cfg["min_cluster_size"]
OPENAI_MODEL        = cfg["openai_model"]
CRS_EPSG            = cfg["crs_epsg"]
LDA_MAX_DF          = cfg["lda_max_df"]
LDA_MIN_DF          = cfg["lda_min_df"]

BUSINESSES_GEOJSON = os.path.join(OUTPUT_DIR, "businesses.geojson")
CLUSTERS_GEOJSON   = os.path.join(OUTPUT_DIR, "clusters.geojson")

# Merge ZIP + demographics
def load_zip_demo() -> gpd.GeoDataFrame:
    zip_gdf = gpd.read_file(ZIP_BOUNDARIES_PATH).to_crs(CRS_EPSG)
    zip_gdf["CODE"] = zip_gdf["CODE"].str.strip()
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")

    pts = zip_gdf.geometry.representative_point()
    zip_gdf["centroid_x"] = pts.x
    zip_gdf["centroid_y"] = pts.y

    df_demo = None
    for fname in ("age.csv", "education.csv", "income.csv"):
        path = os.path.join(DEMOG_BASE_DIR, fname)
        df = pd.read_csv(path)
        df_demo = df.copy() if df_demo is None else df_demo.merge(df, on="Area Code")

    df_demo.columns = df_demo.columns.str.strip().str.lower().str.replace(" ", "_")
    df_demo["area_code"] = df_demo["area_code"].astype(str).str.strip()
    df_demo = df_demo.rename(columns={"area_code": "CODE"})

    merged = zip_gdf.merge(df_demo, on="CODE", how="left")

    for col in merged.columns:
        if col in {"CODE", "geometry", "centroid_x", "centroid_y"}:
            continue
        merged[col] = (
            merged[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"nan": ""})
        )
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged

def precompute_heat(merged_zip: gpd.GeoDataFrame) -> dict:
    exclude = {'CODE','geometry','centroid_x','centroid_y',
        'OBJECTID','COD','Shape__Area','Shape__Length',
        'unnamed:_0_x','age','unnamed:_0_y','unnamed:_0'}
    heat = {}
    for col in merged_zip.columns:
        if col in exclude:
            continue
        vals = merged_zip[col]
        if not vals.notna().any():
            continue
        heat[col] = [
            [float(r.centroid_y), float(r.centroid_x), float(r[col])]
            for _, r in merged_zip.iterrows()
            if pd.notna(r[col])
        ]
    return heat

# Label topics via OpenAI
def label_topic(keywords: list[str], client: OpenAI) -> str:
    prompt = (
        "Below are keywords representing a topic from Yelp reviews:\n"
        f"{', '.join(keywords)}\n\n"
        "Provide a concise, descriptive label for this topic."
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system", "content":"You are a helpful assistant."},
            {"role":"user",   "content":prompt}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Starting precompute pipeline…")

    logging.info("Merging zip boundaries + demographics…")
    zip_demo = load_zip_demo()
    zip_demo.to_file(
        os.path.join(OUTPUT_DIR, "zip_demo.geojson"),
        driver="GeoJSON"
    )

    logging.info("Precomputing heat‑map arrays…")
    heat = precompute_heat(zip_demo)
    with open(os.path.join(OUTPUT_DIR, "heat_data.json"), "w") as fp:
        json.dump(heat, fp)

    if os.path.exists(BUSINESSES_GEOJSON):
        logging.info("Loading existing business centroids…")
        biz_gdf = gpd.read_file(BUSINESSES_GEOJSON)
    else:
        logging.info("Loading reviews parquet…")
        df = pd.read_parquet(INPUT_PARQUET)
        df = df[df["clean_text"].notna()].copy()

        logging.info("Computing sentiment & subjectivity…")
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
        df["sentiment_score"]    = df["clean_text"].apply(lambda t: sia.polarity_scores(t)["compound"])
        df["subjectivity_score"] = df["clean_text"].apply(lambda t: TextBlob(t).sentiment.subjectivity)

        logging.info("Vectorizing text & fitting LDA…")
        reviews    = df["clean_text"].tolist()
        vectorizer = TfidfVectorizer(stop_words="english", max_df=LDA_MAX_DF, min_df=LDA_MIN_DF)
        tfidf      = vectorizer.fit_transform(reviews)
        lda        = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
        lda.fit(tfidf)
        features   = vectorizer.get_feature_names_out()
        topics_kw  = {
            idx: [features[i] for i in comp.argsort()[-10:][::-1]]
            for idx, comp in enumerate(lda.components_)
        }

        logging.info("Labeling topics via OpenAI…")
        dotenv.load_dotenv(override=True)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        topic_labels = {idx: label_topic(kws, client) for idx, kws in topics_kw.items()}

        df["predicted_topic"] = np.argmax(lda.transform(tfidf), axis=1)
        df["topic_label"]     = df["predicted_topic"].map(topic_labels)

        logging.info("Aggregating to business‑level centroids…")
        biz_agg = (
            df.groupby("business_id")
              .agg(
                 latitude          = ("latitude", "first"),
                 longitude         = ("longitude","first"),
                 sentiment_score   = ("sentiment_score","mean"),
                 subjectivity_score= ("subjectivity_score","mean"),
                 stars_rev         = ("stars_rev","mean"),
                 topic_label       = ("topic_label", lambda s: s.mode().iloc[0])
              )
              .reset_index()
        )

        biz_gdf = gpd.GeoDataFrame(
            biz_agg,
            geometry=[Point(xy) for xy in zip(biz_agg.longitude, biz_agg.latitude)],
            crs=CRS_EPSG
        )
        logging.info(f"Saving {BUSINESSES_GEOJSON} ({len(biz_gdf)} points)…")
        biz_gdf.to_file(BUSINESSES_GEOJSON, driver="GeoJSON")

    if os.path.exists(CLUSTERS_GEOJSON):
        logging.info("Loading existing clusters…")
        clusters_gdf = gpd.read_file(CLUSTERS_GEOJSON)
    else:
        logging.info("Running HDBSCAN on ~%d centroids…", len(biz_gdf))
        coords     = biz_gdf[["latitude","longitude"]].to_numpy()
        coords_rad = np.radians(coords)
        clusterer  = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="haversine")
        biz_gdf["cluster_id"] = clusterer.fit_predict(coords_rad)

        logging.info("Computing cluster polygons & metrics…")
        valid = biz_gdf[biz_gdf.cluster_id != -1].copy()

        metrics = (
            valid.groupby("cluster_id")
                 .agg(
                    avg_sentiment    = ("sentiment_score","mean"),
                    avg_subjectivity = ("subjectivity_score","mean"),
                    avg_stars        = ("stars_rev","mean"),
                    business_count   = ("cluster_id","size")
                 )
                 .reset_index()
        )

        dissolved = valid.dissolve(by="cluster_id")
        dissolved["geometry"] = dissolved.geometry.convex_hull
        dissolved = dissolved.reset_index()[["cluster_id", "geometry"]]

        clusters_gdf = dissolved.merge(metrics, on="cluster_id")
        clusters_gdf = gpd.GeoDataFrame(clusters_gdf, geometry="geometry", crs=CRS_EPSG)

        logging.info(f"Saving {CLUSTERS_GEOJSON} ({len(clusters_gdf)} clusters)…")
        clusters_gdf.to_file(CLUSTERS_GEOJSON, driver="GeoJSON")

    logging.info("Precompute pipeline finished. Outputs in `%s`", OUTPUT_DIR)

if __name__ == "__main__":
    main()
