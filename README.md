# CS554 NLP Final Project
NLP Final Project. We are developing a toolkit and application to uncover urban consumer trends and link them to regional socio-economic conditions. 

# Mapping Urban Socio-Economic Trends via Yelp Reviews and Geospatial Analysis

## Overview

This project leverages Yelp review data along with external socio-economic indicators to uncover urban consumer trends and link them to regional socio-economic conditions. Using Natural Language Processing (NLP) techniques, geospatial analysis, and dynamic topic modeling, the goal is to extract meaningful themes from unstructured review texts and understand how they evolve over time.

## Project Objectives
- **NLP Analysis:** Extract and analyze key topics and sentiments from Yelp reviews.
- **Geospatial Clustering:** Map and cluster review data geographically to identify urban hotspots.
- **Socio-Economic Data Integration:** Combine Yelp reviews with demographic and economic data (e.g., from the US Census Bureau) to correlate consumer sentiment with local socio-economic indicators.
- **Actionable Insights:** Provide data-driven recommendations for urban planning, marketing strategies, and business operations.

## Dataset Description

### Yelp Dataset

- **Business Data:** Metadata on businesses (e.g., name, address, star rating, categories, and location).
- **Review Data:** Full review texts, star ratings, and timestamps.
- **User Data:** User profiles, review counts, and additional metadata.

The dataset used in this project is a merged Parquet file combining the above JSON files.

## Data Preprocessing
Preprocessing steps include:
- **Step 1:** Lowercasing
- **Step 2:** Remove Punctuations
- **Step 3:** Tokenization
- **Step 4:** Stop Word Removal
- **Step 5:** Lemmatization
- **Step 6:** Remove Extra Characters(Spaces, Letters, and Special Characters)
- **Step 7:** Emoji -> Text

## Methodology

### 1. Topic Modeling

The project employs a three-pronged approach for Topic Modeling:

- **Static Topic Modeling:**  
  Uses a TF-IDF vectorizer and Latent Dirichlet Allocation (LDA) to extract topics from a sample of reviews. This provides an overall view of common themes in the data.

- **Dynamic Topic Modeling:**  
  Reviews are grouped by year, and a separate LDA model is trained on each time slice. This allows tracking of topic evolution over time.

- **Automatic Topic Labeling:**  
  Instead of manually defining candidate labels, the project leverages a text-to-text generation model to automatically generate concise labels based on the top keywords extracted from each topic.

### 3. Geospatial and Socio-Economic Integration

While the provided code focuses on topic modeling, the overall project includes:
- **Geospatial Analysis:** Clustering businesses and reviews to map urban hotspots.
- **Correlation Analysis:** Merging socio-economic indicators (e.g., median income, education levels, employment rates) with topic trends to derive actionable insights.

## Code Structure

- **Data Loading and Preprocessing:**  
  awofhaeouhwfaiofjawoigjawoijd

- **Topic Modeling:**  
  awofhaeouhwfaiofjawoigjawoijd

- **Geospatial Integration:**  
  awofhaeouhwfaiofjawoigjawoijd

- **Output:**  
  awofhaeouhwfaiofjawoigjawoijd

## How to Run

1. **Install Dependencies:**

  ```bash
  pip install requirements.txt
  ```

2. **Step 2:**
  ????? something

3. **Run the Code:**
  ????? something

## Future Work
- **Enhanced Preprocessing:** Further refine text cleaning, e.g., using custom stopword lists and domain-specific token filtering.
- **Advanced Dynamic Modeling:** Explore dedicated dynamic topic models that jointly capture topic evolution over multiple time slices.
- **Interactive Visualization:** Develop dashboards and interactive maps to visualize topics, sentiment trends, and their correlation with socio-economic variables.

## Acknowledgments
This project builds on publicly available Yelp datasets and leverages SOTA NLP and geospatial analysis techniques to bridge the gap between unstructured review data and structured socio-economic indicators.
