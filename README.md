# XGBoost vs Neural Networks for Airbnb Price Prediction

## Project Overview
This project conducts a detailed comparison between **XGBoost** and two **neural network regression models (NN_v1 and NN_v2)** for predicting **log-transformed Airbnb prices**. The analysis spans **12 U.S. cities**, grouped into three population-based tiers to study performance across different data scales and market complexities.

### City Tiers
- **Large Markets:** New York City, Los Angeles, San Francisco, Chicago  
- **Medium Markets:** Austin, Seattle, Denver, Portland  
- **Small Markets:** Asheville, Santa Cruz County, Salem (OR), Columbus  

The notebook provides a fully automated pipeline covering data acquisition, preprocessing, modeling, evaluation, and cross-tier generalization analysis.

---

## Data Collection and Preprocessing Pipeline

### Automated Data Acquisition
- Scrapes the **InsideAirbnb “Get the Data”** webpage.
- Automatically identifies the most recent `listings.csv.gz` snapshot for each city.
- Downloads and extracts datasets into tier-specific directories:
  - `data/big/`
  - `data/medium/`
  - `data/small/`

### City-Level Data Preparation
- Cleans and standardizes price fields.
- Converts textual bathroom descriptions into numeric values.
- Handles missing values and extreme observations.
- Encodes key categorical attributes.
- Introduces multiple engineered features to improve predictive performance.
- Uses inspection reports and boxplots to validate preprocessing quality before and after cleaning.

---

## Modeling and Evaluation Strategy

### Model Training
- **City-specific models:** XGBoost, NN_v1, NN_v2 trained independently per city.
- **Tier-based models:** Aggregated training for large, medium, and small tiers.
- **Cross-tier experiments:** Composite neural networks trained on one tier and evaluated on others to assess robustness under distribution shift.

### Performance Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

---

## Execution Instructions

### Environment Requirements
The notebook is optimized for **Google Colab** but runs in any Python 3.x environment with the following libraries:

- requests, beautifulsoup4  
- pandas, numpy  
- scikit-learn  
- seaborn, matplotlib  
- xgboost  
- tensorflow / keras  

In Colab, dependencies are installed using:

```bash
pip install requests beautifulsoup4
pip install xgboost shap tensorflow seaborn
```
Running the Notebook
Open raghulch_assignment4.ipynb in Jupyter or Colab and execute cells sequentially:

Downloader: Creates tier-based directories and downloads city datasets.

Preprocessing & Inspection: Applies cleaning pipelines and visual diagnostics.

Training: Trains XGBoost, NN_v1, and NN_v2 per city and reports metrics.

Tier-wise & Cross-tier Analysis: Trains tier-level models and evaluates cross-tier transfer using tables and heatmaps.

All file paths follow a consistent data/*/_listings.csv convention—no manual edits required.

### Data Snapshot Reference
The downloader always selects the most recent available dataset at runtime. For documentation, representative snapshots are summarized below:

| City               | Tier   | Data Snapshot Month | Listings Count |
|-------------------|--------|----------------------|----------------|
| New York City     | Big    | Oct 2025             | 36,111         |
| Los Angeles       | Big    | Sep 2025             | 45,886         |
| San Francisco     | Big    | Sep 2025             | 7,780          |
| Chicago           | Big    | May 2025             | 6,804          |
| Austin            | Medium | May 2025             | 15,187         |
| Seattle           | Medium | Sep 2025             | 6,295          |
| Denver            | Medium | Sep 2025             | 4,910          |
| Portland          | Medium | Sep 2025             | 4,425          |
| Asheville         | Small  | Jun 2025             | 2,876          |
| Santa Cruz County | Small  | Jun 2025             | 1,739          |
| Salem-OR          | Small  | Sep 2025             | 531            |
| Columbus          | Small  | Sep 2025             | 2,877          |

### Preprocessing and Feature Engineering
Numeric Cleaning
Removes currency symbols and converts prices to numeric format.

Drops records with missing prices.

Extracts numeric values from bathroom text fields.

Imputes missing numeric values using medians.

Ensures numeric consistency for capacity, availability, review scores, and stay constraints.

Fills remaining review score gaps with column means.

Categorical Processing
Standardizes room_type and neighbourhood_cleansed.

Retains top 20 neighborhoods; groups remaining values as OTHER.

Applies label encoding to categorical features.

Outlier Treatment
Applies IQR-based winsorization to key numeric attributes.

Reduces extreme values while preserving the bulk of observations.

Uses before/after inspection functions and boxplots to confirm stability.

### Engineered Features
Key engineered attributes include:

price_per_bedroom

avg_review_score

is_entire_home

amenities_count

room_density

review_score_ratio

occupancy_estimate

The final model input combines cleaned numeric features, encoded categoricals, and all engineered variables to predict log(price).

### Key Findings
City-Level Performance
XGBoost consistently achieved the lowest RMSE and MAE and the highest R² across all 12 cities.

Neural networks performed competitively only in data-rich cities.

NN_v1 showed instability in small markets, with strongly negative R² in extreme cases.

NN_v2 improved robustness via Batch Normalization and Dropout but still did not surpass XGBoost.

Tier-Level Results
XGBoost dominated performance across large, medium, and small tiers.

NN_v2 outperformed NN_v1 in most tier-based settings.

Neural models struggled significantly in small-tier markets due to limited data.

### Cross-Tier Generalization Analysis
To study distribution shift, composite neural networks were trained on one tier and evaluated on others:

Large → Medium/Small: Strong generalization, especially with NN_v2.

Medium → Big: NN_v1 failed; NN_v2 retained reasonable performance.

Medium → Small: Both models transferred moderately well.

Small → Medium/Big: Poor performance overall; NN_v2 less severe degradation.

**Conclusion:** NN_v2 generalizes better than NN_v1, but models trained on richer markets transfer best. Small-market data does not scale upward effectively.

### Overall Conclusions
XGBoost is the clear winner, delivering the most accurate and stable results across all cities and tiers.

NN_v2 represents a meaningful architectural improvement over NN_v1 but remains inferior to XGBoost for tabular Airbnb pricing data.

Robust preprocessing and feature engineering are critical for stabilizing both tree-based and neural approaches.

For structured tabular regression tasks, XGBoost remains a powerful baseline, while neural networks require careful design and large, diverse datasets to compete effectively.
