"""
Configuration settings for the Housing Price Prediction API
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data paths
DATA_DIR = BASE_DIR / "data"
HOUSING_DATA_PATH = DATA_DIR / "housing_data.csv"
ZIPCODE_DATA_PATH = DATA_DIR / "zipcode_demographics.csv"

# Model paths
ML_DIR = BASE_DIR / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "housing_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
MODEL_METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_VERSION = "1.0.0"
API_TITLE = "Housing Price Prediction API"
API_DESCRIPTION = "REST API for predicting housing prices with demographic data integration"

# Model Settings
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature columns used by the model
FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms", 
    "sqft_living",
    "sqft_lot",
    "floors",
    "condition",
    "grade",
    "year_built",
    # Demographic features (joined from zipcode data)
    "median_income",
    "population_density",
    "median_age"
]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
