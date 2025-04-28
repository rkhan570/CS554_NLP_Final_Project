# CS554 NLP Final Project: Mapping Urban Socio-Economic Trends via Yelp Reviews and Geospatial Analysis

**Toolkit and application to uncover urban consumer trends and link them to regional socio‑economic conditions in Philadelphia**. By combining Yelp review data with demographic indicators and geospatial analysis, this project delivers actionable insights through a dynamic topic modeling pipeline and an interactive Streamlit dashboard.

---

## Overview

- Leverage Yelp business, review, and user data alongside external socio‑economic indicators (age, income, education by ZIP code).
- Apply NLP for sentiment analysis and LDA topic modeling (static & dynamic).
- Perform geospatial clustering (HDBSCAN) to identify urban hotspots of consumer trends.
- Integrate demographic heatmaps and cluster metrics in a Folium-based Streamlit dashboard.

---

## Project Objectives

1. **NLP Analysis**: Preprocess review text (lowercase, punctuation removal, tokenization, stop-word removal, lemmatization, emoji conversion), compute sentiment & subjectivity scores, and extract topics via LDA.
2. **Dynamic Topic Modeling**: Train LDA per year slice to track topic evolution.
3. **Automatic Topic Labeling**: Use OpenAI’s GPT to generate concise labels from top keywords.
4. **Geospatial Clustering**: Cluster businesses with HDBSCAN and compute cluster-level metrics (avg sentiment, subjectivity, star rating, count).
5. **Socio-Economic Integration**: Merge ZIP-code boundaries with Census-derived demographics to create choropleth and heatmap layers.
6. **Interactive Dashboard**: Streamlit + Folium map enabling variable selection, marker/cluster toggles, and detailed tooltips/popups.

---

## Dataset Description

- **reviews_parquet**: Merged Parquet of Yelp business, review, and user JSON files.
- **Zip Boundaries**: GeoJSON of Philadelphia ZIP-code polygons.
- **Demographic CSVs**: Age, education, and income data per ZIP code.

---

## Methodology

### 1. Data Preprocessing
- Lowercasing, punctuation removal, tokenization, stop-word removal, lemmatization, cleaning special characters, converting emojis to text.

### 2. Topic Modeling
- **Static**: TF-IDF + LDA (n topics) on full corpus.
- **Dynamic**: Year grouped LDA models to capture temporal shifts.
- **Labeling**: Call OpenAI GPT for topic names from keyword lists.

### 3. Geospatial Analysis
- **Demographic Merge**: Load ZIP shapefile, compute centroids, merge with demographic CSVs, and prepare heat arrays.
- **Business Aggregation**: Compute business-level sentiment & topic centroids.
- **HDBSCAN Clustering**: Cluster centroid coordinates (haversine metric), dissolve to convex hull polygons, and calculate cluster metrics.

### 4. Dashboard
- Folium choropleth & heatmap layers for demographics.
- MarkerCluster for businesses colored by sentiment.
- GeoJSON overlay for clusters with hover tooltips and popups.

---

## Repository Structure

```
├── precompute.py           # Batch pipeline: demographics merge, heat data, NLP, topics, clusters
├── dashboard/
│   └── app.py              # Streamlit dashboard loading precomputed outputs
├── config.yaml             # I/O paths & parameters: datasets, NLP, clustering, map defaults
├── requirements.txt        # Python dependencies
├── .env                    # Env vars (OPENAI_API_KEY)
├── precomputed/            # Generated outputs:
│   ├── zip_demo.geojson    
│   ├── heat_data.json      
│   ├── businesses.geojson  
│   └── clusters.geojson
├── notebooks/              # .ipynb Notebooks showing for each NLP phase and validation
│   ├── data_cleaning/      # Code to clean our data
│   ├── geoclustering/      # Code for geoclustering
│   ├── sentiment_analysis/ # Code for sentiment_analysis
│   ├── topic_modeling/     # Code for topic modeling
│   └── validation/         # Code for validation: clustering, correlation, sentiment_analysis, topic_modeling
└── datasets/               # Raw inputs:
    ├── demographic_infused_philly_c.parquet
    ├── Philly_boundaries/Zipcodes_Poly.geojson
    └── demographic_dataset/{age.csv,education.csv,income.csv}
```

---

## Prerequisites

- Python 3.11+  
- GDAL, GEOS, PROJ system libs (GeoPandas)  
- OPENAI_API_KEY for automatic labeling

---

## Installation

1. Clone & enter repo:
   ```bash
   git clone https://github.com/rkhan570/CS554_NLP_Final_Project.git
   cd CS554_NLP_Final_Project
   ```
2. Create & activate venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up `.env`:
   ```dotenv
   OPENAI_API_KEY=sk-...
   ```

---

## Configuration (`config.yaml`)

```yaml
reviews_parquet: "./datasets/demographic_infused_philly_c.parquet"
precompute_dir: "precomputed"
zip_boundaries: "./datasets/Philly_boundaries/Zipcodes_Poly.geojson"
demog_base_dir: "./datasets/demographic_dataset"

num_topics:       10
min_cluster_size: 25
openai_model:     "gpt-4o-mini"
crs_epsg:         "EPSG:4326"
lda_max_df:       0.95
lda_min_df:       10
map_center:       [39.9526, -75.1652]
map_zoom:         12
```  

---

## How to Run

1. **Precompute**:
   ```bash
   python precompute.py
   ```
2. **Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

---

## Future Work

- Advanced dynamic topic modeling frameworks.  
- Additional socio-economic layers (unemployment, housing).  
- Web deployment with user management.

---

## Acknowledgments

Built on Yelp Open Dataset and U.S. Census data; leverages GeoPandas, HDBSCAN, and OpenAI for labeling.

---

## License

MIT License

---