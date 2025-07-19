#  Demand Forecasting for Smart Business

##  Project Overview
Brazilian e-commerce demand forecasting project that transforms extremely sparse daily product data (99.6% zeros) into accurate category-level predictions through smart data aggregation and modeling.

##  Key Achievement
**Final Model Performance: R² = 0.832** (83% variance explained) - Production-ready demand forecasting system!

##  Data Journey & Results
- **Starting Point**: 23.6M daily product-date combinations (99.6% sparsity)
- **Stage 1**: Daily → Weekly aggregation (7x reduction)
- **Stage 2**: Product → Category aggregation (455x total reduction)
- **Final Dataset**: 7,384 category-week records with robust patterns
- **Performance Improvement**: Impossible → R² = 0.832

##  Notebooks (Sequential Order)

### 00_data_exploration.ipynb
**Daily Product-Level Data Exploration**
- Explores 23.6M product-date combinations
- Identifies 99.6% sparsity problem
- Analyzes 71 product categories, 32K products
- Discovers need for aggregation strategy

### 01_daily_product_baseline.ipynb
**Daily Product-Level Baseline Modeling**
- Attempts modeling at daily product level
- Creates lag features and time series features
- Results: Poor performance due to extreme sparsity
- Confirms daily level is too sparse for reliable forecasting

### 02_weekly_product_modeling.ipynb
**Weekly Product-Level Modeling**
- Transforms daily → weekly data (23.6M → 3.4M records)
- Engineering: lag features, rolling means, price features
- Results: Still challenging due to product-level noise
- Prepares data for category-level aggregation

### 03_category_level_forecasting.ipynb ⭐
**Category-Level Forecasting (FINAL MODEL)**
- Aggregates product-week → category-week (455x reduction!)
- Global CatBoost model across all 71 categories
- **Results: R² = 0.832, MAE ≈ 6 units**
- Production-ready 2-week ahead forecasting
- Model comparison: CatBoost vs LightGBM vs XGBoost vs RandomForest
- Cross-validation with TimeSeriesSplit
- Feature importance analysis
- Model persistence for deployment

## 🏗️ Project Structure
```
forecasting_demand_smart_business/
├── 📓 notebooks/
│   ├── 00_data_exploration.ipynb          # Data exploration
│   ├── 01_daily_product_baseline.ipynb    # Daily modeling attempts
│   ├── 02_weekly_product_modeling.ipynb   # Weekly transformation
│   └── 03_category_level_forecasting.ipynb # Final solution ⭐
├── 📁 data/                               # Raw CSV files
├── 📈 outputs/                            # Processed datasets
├── 🧠 models/                             # Trained models
└── 📋 requirements.txt                    # Dependencies
```

##  Key Insights

### ✅ **What Worked**
- **Smart Aggregation**: Product → Category level solved sparsity
- **Global Modeling**: One model for all categories (vs 71 separate models)
- **Time Series Features**: Lag features and rolling means were most important
- **CatBoost**: Best performing algorithm for this dataset

### ❌ **What Didn't Work**
- **Daily Product Level**: 99.6% sparsity made modeling impossible
- **Weekly Product Level**: Still too noisy and sparse
- **Category-Specific Models**: Insufficient data per category

### **Final Model Features (By Importance)**
1. `rolling_mean_4w` (29.5%) - Most important
2. `lag_2w` (13.0%) 
3. `lag_1w` (12.9%)
4. `active_products` (9.7%)
5. `month_sin` (9.7%)
6. `category` (1.5%) - Low importance validates global approach
``

## Business Impact
- **83% accuracy** enables reliable inventory planning
- **2-week forecast horizon** allows proactive supply chain decisions
- **Single model** handles all 71 categories consistently
- **Production-ready** system with saved model artifacts

## Technical Stack
- **Python**: pandas, numpy, scikit-learn
- **ML Models**: CatBoost, LightGBM, XGBoost, RandomForest  
- **Validation**: TimeSeriesSplit cross-validation
- **Deployment**: Joblib model persistence

## Performance Summary
| Metric | Value | Interpretation |
|--------|--------|----------------|
| R² Score | 0.832 | Excellent - explains 83% of variance |
| MAE | ~6 units | Average prediction error |
| RMSLE | 0.691 | Good for demand forecasting |
| Categories | 71 | All handled by single model |

** This project demonstrates how thoughtful data aggregation can transform impossible forecasting problems into production-ready solutions!**

Author: Natalia Marko