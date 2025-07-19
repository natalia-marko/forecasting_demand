"""
utiles.py - Comprehensive Utilities for Demand Forecasting Project
=======================================================================

This module contains utility functions for data loading, preprocessing, 
feature engineering, model evaluation, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Plotting configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_demand_data(filepath='../data/demand_data_pp.csv'):
    """Load and prepare demand data with proper datetime parsing."""
    try:
        df = pd.read_csv(filepath)
        if 'order_purchase_timestamp' in df.columns:
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        print(f"✓ Loaded {len(df)} records from {filepath}")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def load_all_data():
    """Load all relevant datasets for analysis."""
    data_files = {
        'demand': '../data/demand_data_pp.csv',
        'customers': '../data/customers.csv',
        'orders': '../data/orders.csv',
        'products': '../data/products.csv',
        'categories': '../data/category_data.csv'
    }
    
    datasets = {}
    for name, filepath in data_files.items():
        try:
            datasets[name] = pd.read_csv(filepath)
            print(f"✓ Loaded {name}: {len(datasets[name])} records")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    return datasets


def remove_outliers(df, exclude_from_outlier_removal=['daily_demand']):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_cols = [col for col in numeric_cols if col not in exclude_from_outlier_removal]
    outliers_removed = 0
    
    for col in outlier_cols:
        if col in df.columns:
            initial_count = len(df)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            outliers_removed += (initial_count - len(df))
    
    print(f"Removed {outliers_removed} outlier rows from feature columns")
    return df

##================================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_demand_dashboard(data, date_col='order_date', demand_col='daily_demand', category_col='category', dashboard_title='Brazilian E-commerce Demand Analysis Dashboard'):
    """
    Creates a 4-panel dashboard visualizing demand patterns.
    
    Parameters:
        data (pd.DataFrame): The dataset containing time series demand.
        date_col (str): Column name for dates.
        demand_col (str): Column name for demand/sales.
        category_col (str): Column name for product categories.
        dashboard_title (str): Title of the entire dashboard.
    """
    print("\n CREATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 60)

    # Ensure datetime format
    data[date_col] = pd.to_datetime(data[date_col])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(dashboard_title, fontsize=16, fontweight='bold')

    # ───── PLOT 1: Daily Demand Over Time ─────
    print("Plotting daily demand over time...")
    daily_demand = data.groupby(date_col)[demand_col].sum()
    axes[0, 0].plot(daily_demand.index, daily_demand.values, linewidth=1, alpha=0.8, color='#1f77b4')
    axes[0, 0].set_title('Total Daily Demand Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Total Demand')
    axes[0, 0].grid(True, alpha=0.3)
    for tick in axes[0, 0].get_xticklabels():
        tick.set_rotation(45)

    # Add linear trend line
    z = np.polyfit(range(len(daily_demand)), daily_demand.values, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(daily_demand.index, p(range(len(daily_demand))), "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[0, 0].legend()

    # ───── PLOT 2: Demand Distribution (Non-zero only) ─────
    print("Plotting demand distribution (non-zero)...")
    non_zero_demand = data[data[demand_col] > 0][demand_col]
    axes[0, 1].hist(non_zero_demand, bins=50, alpha=0.7, edgecolor='black', color='#ff7f0e')
    axes[0, 1].set_title('Demand Distribution (Non-Zero Values Only)', fontweight='bold')
    axes[0, 1].set_xlabel('Demand')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    mean_demand = non_zero_demand.mean()
    median_demand = non_zero_demand.median()
    axes[0, 1].axvline(mean_demand, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_demand:.2f}')
    axes[0, 1].axvline(median_demand, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_demand:.2f}')
    axes[0, 1].legend()

    # ───── PLOT 3: Top Categories ─────
    print("Plotting top product categories...")
    top_categories = data.groupby(category_col)[demand_col].sum().sort_values(ascending=False).head(15)
    y_pos = range(len(top_categories))
    bars = axes[1, 0].barh(y_pos, top_categories.values, color='#2ca02c', alpha=0.8)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels([cat[:20] + '...' if len(cat) > 20 else cat for cat in top_categories.index])
    axes[1, 0].set_xlabel('Total Demand')
    axes[1, 0].set_title('Top 15 Product Categories by Total Demand', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[1, 0].text(width + max(top_categories.values) * 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.0f}', ha='left', va='center', fontsize=8)

    # ───── PLOT 4: Monthly Demand ─────
    print("Plotting monthly demand patterns...")
    data['month'] = data[date_col].dt.month
    monthly_demand = data.groupby('month')[demand_col].sum()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bars = axes[1, 1].bar(range(1, 13), monthly_demand.values, color='#d62728', alpha=0.8)
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].set_xticklabels(month_names)
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Total Demand')
    axes[1, 1].set_title('Monthly Demand Distribution', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(monthly_demand.values) * 0.01,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for title
    plt.show()

    print("Key patterns identified for feature engineering")


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def create_time_features(df, date_col='order_purchase_timestamp'):
    """Create time-based features from datetime column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Basic time features
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['weekday'] = df[date_col].dt.weekday
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    
    # Seasonal features
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    print(f"✓ Created {len([col for col in df.columns if col.endswith(('_sin', '_cos')) or col in ['year', 'month', 'day', 'weekday', 'week_of_year', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end']])} time features")
    return df



def create_price_features(df):
    """Create price-based features"""
    df = df.copy()
    df['price_log'] = np.log1p(df['price'])

    df['price_rank_in_category'] = df.groupby('categ')['price'].rank(pct=True)
    df['price_vs_category_mean'] = df['price'] / df.groupby('categ')['price'].transform('mean')
    
    print("✓ Created price-based features")
    return df

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def calculate_rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error."""
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Comprehensive model evaluation."""
    metrics = {
        'Model': model_name,
        'RMSLE': calculate_rmsle(y_true, y_pred),
        'RMSE': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'MAE': np.mean(np.abs(y_true - y_pred)),
        'MAPE': calculate_mape(y_true, y_pred),
        'SMAPE': calculate_smape(y_true, y_pred),
        'R2': np.corrcoef(y_true, y_pred)[0, 1] ** 2 if len(y_true) > 1 else 0
    }
    
    return metrics

def compare_models(results_dict):
    """Compare multiple models and return sorted results."""
    comparison_df = pd.DataFrame(results_dict).T
    comparison_df = comparison_df.sort_values('RMSLE')
    
    print("\n" + "="*60)
    print("MODEL COMPARISON (sorted by RMSLE)")
    print("="*60)
    print(comparison_df.round(4))
    
    return comparison_df

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_time_series(df, date_col='order_purchase_timestamp', target_col='target_2w', 
                    title="Time Series Plot", figsize=(12, 6)):
    """Plot time series data."""
    plt.figure(figsize=figsize)
    plt.plot(pd.to_datetime(df[date_col]), df[target_col], linewidth=1)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, title="Predictions vs Actual", figsize=(10, 6)):
    """Plot predictions vs actual values."""
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, title="Residuals Plot", figsize=(10, 6)):
    """Plot residuals."""
    residuals = y_true - y_pred
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importance_dict, title="Feature Importance", top_n=20, figsize=(10, 8)):
    """Plot feature importance."""
    if isinstance(importance_dict, dict):
        features = list(importance_dict.keys())[:top_n]
        importance = list(importance_dict.values())[:top_n]
    else:
        features = importance_dict['feature'][:top_n]
        importance = importance_dict['importance'][:top_n]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(features)), importance)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def save_plot(filename, output_dir='../outputs/'):
    """Save the current plot."""
    plt.savefig(f'{output_dir}{filename}', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved as {output_dir}{filename}")

# ============================================================================
# DATA ANALYSIS HELPERS
# ============================================================================

def data_overview(df, title="Dataset Overview"):
    """Print comprehensive data overview."""
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("\nColumn types:")
    print(df.dtypes.value_counts())
    print("\nNumerical columns summary:")
    print(df.describe().round(2))

def category_analysis(df, category_col='categ', target_col='target_2w'):
    """Analyze target variable by category."""
    if category_col not in df.columns:
        print(f"Column {category_col} not found")
        return
    
    analysis = df.groupby(category_col)[target_col].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    
    analysis = analysis.sort_values('mean', ascending=False)
    print("\n" + "="*60)
    print(f"TARGET ANALYSIS BY {category_col.upper()}")
    print("="*60)
    print(analysis)
    
    return analysis

def correlation_analysis(df, target_col='target_2w', threshold=0.1):
    """Analyze correlations with target variable."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
    
    high_corr = correlations[correlations > threshold]
    print(f"\nFeatures with correlation > {threshold}:")
    print(high_corr.round(3))
    
    return high_corr

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def memory_usage(df):
    """Check memory usage of DataFrame."""
    memory_gb = df.memory_usage(deep=True).sum() / 1024**3
    print(f"DataFrame memory usage: {memory_gb:.3f} GB")
    return memory_gb

def reduce_memory_usage(df):
    """Reduce memory usage by optimizing dtypes."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df

def print_separator(title="", char="=", width=60):
    """Print a formatted separator."""
    if title:
        print(f"\n{char * width}")
        print(f"{title.center(width)}")
        print(f"{char * width}")
    else:
        print(f"{char * width}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default parameters for models
DEFAULT_CATBOOST_PARAMS = {
    'iterations': 300,
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 30,
    'use_best_model': True
}

DEFAULT_LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

# ============================================================================
# FEATURE ENGINEERING - SIMPLIFIED
# ============================================================================

def create_lag_features(df, target_col='demand', lags=[1, 7]):
    """Create simple lagged features - just yesterday and last week."""
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    print(f"✓ Created {len(lags)} lag features")
    return df

def create_rolling_features(df, target_col='demand', windows=[7]):
    """Create simple rolling features - just 7-day average."""
    df = df.copy()
    
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
    
    print(f"✓ Created {len(windows)} rolling features")
    return df

# Feature groups for easy selection
FEATURE_GROUPS = {
    'time_features': ['year', 'month', 'day', 'weekday', 'week_of_year', 'quarter',
                     'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
                     'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'],
    'lag_features': ['demand_lag_1', 'demand_lag_7'],
    'rolling_features': ['demand_rolling_mean_7'],
    'price_features': ['price', 'price_log', 'price_rank_in_category', 'price_vs_category_mean']
}
