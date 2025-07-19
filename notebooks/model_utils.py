"""
Model utilities for the Forecasting Demand Smart Business project.
Contains functions for training, evaluating, and managing ML models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import sys

# ML models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Time series
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Text processing
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import re

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

def prepare_features(df: pd.DataFrame, target_col: str = 'demand', 
                    exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for ML model training.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        exclude_cols: Columns to exclude from features
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if exclude_cols is None:
        exclude_cols = ['order_date', 'product_id', 'categ']
    
    # Get feature columns
    feature_cols = [col for col in df.columns 
                   if col != target_col and col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values in features
    X = X.fillna(X.mean())
    
    print(f"  📊 Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"  🎯 Target: {target_col} (range: {y.min():.1f} - {y.max():.1f})")
    
    return X, y

def train_forecasting_models(X: pd.DataFrame, y: pd.Series, 
                           test_size: float = None) -> Dict[str, Any]:
    """
    Train multiple forecasting models and compare performance.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Test set size (defaults to config)
        
    Returns:
        Dictionary containing trained models and metrics
    """
    if test_size is None:
        test_size = FORECAST_PARAMS['test_size']
    
    print("🤖 Training forecasting models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, 
        random_state=NOTEBOOK_PARAMS['random_seed']
    )
    
    # Initialize models
    models = {
        'CatBoost': CatBoostRegressor(**MODEL_PARAMS['catboost']),
        'XGBoost': XGBRegressor(**MODEL_PARAMS['xgboost']),
        'LightGBM': LGBMRegressor(**MODEL_PARAMS['lightgbm']),
        'RandomForest': RandomForestRegressor(**MODEL_PARAMS['random_forest'])
    }
    
    results = {
        'models': {},
        'metrics': {},
        'predictions': {}
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"  Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        try:
            metrics['train_mape'] = mean_absolute_percentage_error(y_train, y_pred_train)
            metrics['test_mape'] = mean_absolute_percentage_error(y_test, y_pred_test)
        except:
            metrics['train_mape'] = np.nan
            metrics['test_mape'] = np.nan
        
        results['models'][name] = model
        results['metrics'][name] = metrics
        results['predictions'][name] = {
            'y_test': y_test,
            'y_pred': y_pred_test
        }
        
        print(f"    ✅ {name}: Test MAE = {metrics['test_mae']:.2f}, R² = {metrics['test_r2']:.3f}")
    
    # Add data splits to results
    results['data_splits'] = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test
    }
    
    return results

def evaluate_model_performance(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create comprehensive model performance comparison.
    
    Args:
        results: Results from train_forecasting_models
        
    Returns:
        DataFrame with model performance metrics
    """
    metrics_data = []
    
    for model_name, metrics in results['metrics'].items():
        metrics_data.append({
            'Model': model_name,
            'Train_MAE': metrics['train_mae'],
            'Test_MAE': metrics['test_mae'],
            'Train_MSE': metrics['train_mse'],
            'Test_MSE': metrics['test_mse'],
            'Train_R2': metrics['train_r2'],
            'Test_R2': metrics['test_r2'],
            'Train_MAPE': metrics.get('train_mape', np.nan),
            'Test_MAPE': metrics.get('test_mape', np.nan),
            'Overfit_Score': metrics['train_mae'] - metrics['test_mae']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.sort_values('Test_MAE')
    
    print("📊 Model Performance Comparison:")
    print(metrics_df[['Model', 'Test_MAE', 'Test_R2', 'Overfit_Score']].to_string(index=False))
    
    return metrics_df

def create_forecasts(model: Any, last_data: pd.DataFrame, 
                    forecast_days: int = None) -> pd.DataFrame:
    """
    Create future forecasts using trained model.
    
    Args:
        model: Trained ML model
        last_data: Last available data for feature engineering
        forecast_days: Number of days to forecast
        
    Returns:
        DataFrame with forecasts
    """
    if forecast_days is None:
        forecast_days = FORECAST_PARAMS['forecast_days']
    
    print(f"🔮 Creating {forecast_days}-day forecasts...")
    
    # Get last date
    last_date = last_data['order_date'].max()
    start_date = last_date + timedelta(days=FORECAST_PARAMS['start_days_after'])
    
    forecast_dates = [
        start_date + timedelta(days=i) for i in range(forecast_days)
    ]
    
    forecasts = []
    
    for category in last_data['categ'].unique():
        cat_data = last_data[last_data['categ'] == category].tail(30)  # Last 30 days
        
        if len(cat_data) == 0:
            continue
            
        last_row = cat_data.iloc[-1]
        
        for forecast_date in forecast_dates:
            # Create features for forecast (simplified)
            features = {
                'price': last_row['price'],
                'day_of_week': forecast_date.weekday(),
                'month': forecast_date.month,
                'quarter': (forecast_date.month - 1) // 3 + 1,
                'week_of_year': forecast_date.isocalendar()[1],
                'is_weekend': int(forecast_date.weekday() >= 5),
                'lag_1': last_row.get('demand', 1),
                'lag_2': cat_data.iloc[-2].get('demand', 1) if len(cat_data) >= 2 else last_row.get('demand', 1),
                'lag_3': cat_data.iloc[-3].get('demand', 1) if len(cat_data) >= 3 else last_row.get('demand', 1),
                'lag_7': cat_data.iloc[-7].get('demand', 1) if len(cat_data) >= 7 else last_row.get('demand', 1),
                'rolling_mean_7': cat_data.tail(7)['demand'].mean() if len(cat_data) >= 7 else last_row.get('demand', 1),
                'rolling_std_7': cat_data.tail(7)['demand'].std() if len(cat_data) >= 7 else 0,
            }
            
            # Add more rolling features if available
            for window in [14, 30]:
                if len(cat_data) >= window:
                    features[f'rolling_mean_{window}'] = cat_data.tail(window)['demand'].mean()
                    features[f'rolling_std_{window}'] = cat_data.tail(window)['demand'].std()
                    features[f'rolling_max_{window}'] = cat_data.tail(window)['demand'].max()
                    features[f'rolling_min_{window}'] = cat_data.tail(window)['demand'].min()
                else:
                    features[f'rolling_mean_{window}'] = features['rolling_mean_7']
                    features[f'rolling_std_{window}'] = features['rolling_std_7']
                    features[f'rolling_max_{window}'] = features['lag_1']
                    features[f'rolling_min_{window}'] = features['lag_1']
            
            # Create feature vector
            X_forecast = pd.DataFrame([features])
            
            # Make prediction
            try:
                prediction = model.predict(X_forecast)[0]
                prediction = max(0, round(prediction))  # Ensure non-negative integer
            except:
                prediction = 1  # Fallback
            
            forecasts.append({
                'categ': category,
                'forecast_date': forecast_date,
                'predicted_demand': prediction
            })
    
    forecast_df = pd.DataFrame(forecasts)
    print(f"  ✅ Created forecasts for {len(forecast_df)} category-date combinations")
    
    return forecast_df

def train_sarimax_model(ts_data: pd.Series, seasonal_period: int = 7) -> Tuple[Any, np.ndarray]:
    """
    Train SARIMAX model for time series forecasting.
    
    Args:
        ts_data: Time series data
        seasonal_period: Seasonal period (default: 7 for weekly)
        
    Returns:
        Tuple of (fitted model, forecast)
    """
    print("📈 Training SARIMAX model...")
    
    try:
        # Fit SARIMAX model
        model = SARIMAX(
            ts_data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_period)
        )
        fitted_model = model.fit(disp=False)
        
        # Create forecast
        forecast = fitted_model.forecast(steps=FORECAST_PARAMS['forecast_days'])
        forecast = np.maximum(0, np.round(forecast)).astype(int)
        
        print(f"  ✅ SARIMAX model trained. AIC: {fitted_model.aic:.2f}")
        print(f"  🔮 Forecast range: {forecast.min()} - {forecast.max()}")
        
        return fitted_model, forecast
        
    except Exception as e:
        print(f"  ❌ SARIMAX training failed: {e}")
        return None, None

def train_sentiment_model(reviews_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train sentiment analysis model for customer reviews.
    
    Args:
        reviews_df: DataFrame with review text and scores
        
    Returns:
        Dictionary with trained model and results
    """
    print("💬 Training sentiment analysis model...")
    
    # Prepare data
    df = reviews_df[['review_score', 'review_comment_title', 'review_comment_message']].copy()
    
    # Map scores to sentiment
    def map_sentiment(score):
        if score in TEXT_PARAMS['sentiment_mapping']['positive']:
            return 'positive'
        elif score in TEXT_PARAMS['sentiment_mapping']['neutral']:
            return 'neutral'
        else:
            return 'negative'
    
    df['sentiment'] = df['review_score'].apply(map_sentiment)
    
    # Preprocess text
    df['review_comment_title'] = df['review_comment_title'].fillna('').astype(str)
    df['review_comment_message'] = df['review_comment_message'].fillna('').astype(str)
    df['merged_review'] = df['review_comment_title'] + ' ' + df['review_comment_message']
    
    # Clean text
    df['merged_review'] = df['merged_review'].str.lower()
    df['merged_review'] = df['merged_review'].str.replace(r'[^a-zA-Z0-9\sáéíóúàèìòùâêîôûãõç]', '', regex=True)
    df['merged_review'] = df['merged_review'].str.replace(r'\s+', ' ', regex=True)
    
    # Remove empty reviews
    df = df[df['merged_review'].str.strip() != '']
    
    # Download NLTK data if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    # Vectorize text
    stop_words = list(stopwords.words(TEXT_PARAMS['language']))
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=TEXT_PARAMS['max_features'],
        ngram_range=TEXT_PARAMS['ngram_range']
    )
    
    X = vectorizer.fit_transform(df['merged_review'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=NOTEBOOK_PARAMS['random_seed'], stratify=y
    )
    
    # Train model
    model = LogisticRegression(random_state=NOTEBOOK_PARAMS['random_seed'], max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    
    print(f"  ✅ Sentiment model trained. Accuracy: {accuracy:.3f}")
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'processed_data': df,
        'predictions': y_pred,
        'test_labels': y_test
    }

def extract_prices_from_text(text_series: pd.Series) -> List[List[str]]:
    price_pattern = TEXT_PARAMS['price_regex']
    
    prices = []
    for text in text_series:
        matches = re.findall(price_pattern, str(text).lower())
        if matches:
            # Flatten the tuple results and filter out empty strings
            flat_matches = [item for sublist in matches for item in sublist if item]
            prices.append(flat_matches)
        else:
            prices.append([])
    
    return prices

def save_models(models: Dict[str, Any], model_dir: Path = None) -> Dict[str, Path]:

    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "models"
    
    model_dir.mkdir(exist_ok=True)
    saved_paths = {}
    
    for model_name, model in models.items():
        file_path = model_dir / f"{model_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, file_path)
        saved_paths[model_name] = file_path
        print(f"  💾 Saved {model_name} to {file_path}")
    
    return saved_paths

def load_models(model_paths: Dict[str, Path]) -> Dict[str, Any]:
    loaded_models = {}
    
    for model_name, file_path in model_paths.items():
        if file_path.exists():
            loaded_models[model_name] = joblib.load(file_path)
            print(f"  📂 Loaded {model_name} from {file_path}")
        else:
            print(f"  ⚠️  {model_name}: File not found at {file_path}")
    
    return loaded_models

def calculate_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:

    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    else:
        print("  ⚠️  Model does not have feature_importances_ attribute")
        return pd.DataFrame()

def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:

    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    
    return {
        'cv_mae_mean': -cv_scores.mean(),
        'cv_mae_std': cv_scores.std(),
        'cv_scores': -cv_scores
    } 

import matplotlib.pyplot as plt

import numpy as np

def plot_prediction_dashboard(y_test, y_pred, title="Prediction Analysis Dashboard"):
    """Create a comprehensive 2x2 dashboard for prediction analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Predicted vs Actual
    axes[0,0].scatter(y_test, y_pred, alpha=0.6, s=30, color='blue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0,0].set_xlabel('Actual Demand')
    axes[0,0].set_ylabel('Predicted Demand')
    axes[0,0].set_title('Predicted vs Actual')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_test, y_pred)
    axes[0,0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0,0].transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    # 2. Residual Plot
    residuals = y_test - y_pred
    axes[0,1].scatter(y_pred, residuals, alpha=0.6, s=30, color='green')
    axes[0,1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0,1].set_xlabel('Predicted Demand')
    axes[0,1].set_ylabel('Residuals (Actual - Predicted)')
    axes[0,1].set_title('Residual Plot')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Error Distribution
    axes[1,0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Error Distribution')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Performance Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # Calculate RMSLE safely
    def rmsle(actual, predicted):
        # Add 1 to avoid log(0)
        log_diff = np.log1p(predicted) - np.log1p(actual)
        return np.sqrt(np.mean(log_diff ** 2))
    
    rmsle_val = rmsle(y_test, y_pred)
    
    metrics = ['MAE', 'MSE', 'RMSLE', 'R²']
    values = [mae, mse, rmsle_val, r2]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    bars = axes[1,1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Performance Metrics')
    axes[1,1].set_ylabel('Value')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
