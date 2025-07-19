Certainly! Here’s a **KISS-style README.md** for your project, written from the perspective of a data scientist, reflecting your task requirements and your data’s characteristics.

# Demand Forecasting for E-commerce Categories

## Project Goal

Develop two demand forecasting models for all product groups:
- **A machine learning model**
- **A classical time series model**

Forecast horizon: **14 days ahead, with a 7-day gap from the last available date**.

## Data Overview

- **Granularity:** Product-level, daily
- **Columns:** `order_date`, `price`, `demand`
- **Size:** ~23.6 million rows
- **Sparsity:** Most `demand` values are zero

## Workflow

### 1. Data Preparation

- **Aggregate:** Convert product-level daily data to **product-level weekly data** weekly_data_item.csv
- **Aggregate:** Convert product-level daily data to **category-level weekly data** weekly_data_categ.csv
- **Feature Engineering:**  
  - Temporal: lags, rolling means, trends  
  - Price: average, changes, volatility  
  - Calendar: week, month, seasonality  
  - Target: total demand per group for the forecast period

### 2. Model 1: Machine Learning

- **Model:** Gradient Boosting (e.g., LightGBM or XGBoost)
- **Why:**  
  - Handles sparse data well  
  - Captures non-linear effects  
  - Robust to missing/zero values
- **Features:** All engineered features
- **Validation:** Time series cross-validation

### 3. Model 2: Classical Time Series

- **Model:** Exponential Smoothing (per category)
- **Why:**  
  - Standard for univariate series  
  - Interpretable, baseline for comparison
- **Features:** Only historical demand
- **Validation:** Walk-forward validation

### 4. Forecasting

- **Scoring:** Both models should output forecasts for each group, for the required future period.
- **Evaluation:** Compare models using MAE/RMSE on a holdout period.

## Deliverables

- **Notebook (.ipynb):** Contains all code, explanations, and scoring interface.
- **README.md:** This file.

## Comments on Method Choice

- **ML model** is chosen for flexibility and ability to use rich features.
- **Classical model** serves as a transparent, interpretable baseline.
- Both approaches are standard in demand forecasting, and together provide robust, complementary results.

## How to Use

1. Place your raw data in the `data/` folder.
2. Run the notebook step by step.
3. At the end, use the scoring cell to generate forecasts for any future period.

## Keep It Simple

- All code is modular and easy to follow.
- No unnecessary complexity—focus on practical, proven methods.
- Clear explanations for every step.

