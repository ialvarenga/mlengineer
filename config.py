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

# AWS Settings
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_SECRET_NAME = os.getenv("AWS_SECRET_NAME", "phdata-demoapp")
USE_AWS_SECRETS = os.getenv("USE_AWS_SECRETS", "true").lower() == "true"

# JWT Settings (loaded from AWS Secrets Manager if USE_AWS_SECRETS=true)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY is required. Set it via environment variable or AWS Secrets Manager.")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "30"))

# User credentials (loaded from AWS Secrets Manager if USE_AWS_SECRETS=true)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not ADMIN_USERNAME or not ADMIN_PASSWORD:
    raise ValueError("ADMIN_USERNAME and ADMIN_PASSWORD are required. Set them via environment variables or AWS Secrets Manager.")
