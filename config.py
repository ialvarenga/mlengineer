"""
Configuration settings for the Housing Price Prediction API
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data paths
DATA_DIR = BASE_DIR / "data"
HOUSING_DATA_PATH = DATA_DIR / "kc_house_data.csv"
ZIPCODE_DATA_PATH = DATA_DIR / "zipcode_demographics.csv"
FUTURE_EXAMPLES_PATH = DATA_DIR / "future_unseen_examples.csv"

# Model paths (created by create_model.py)
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
MODEL_FEATURES_PATH = MODEL_DIR / "model_features.json"

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_VERSION = "1.0.0"
API_TITLE = "Housing Price Prediction API"
API_DESCRIPTION = (
    "REST API for predicting housing prices with demographic data integration"
)

# Model Settings
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# AWS Settings
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_SECRET_NAME = os.getenv("AWS_SECRET_NAME", "phdata-demoapp")
USE_AWS_SECRETS = os.getenv("USE_AWS_SECRETS", "true").lower() == "true"

# S3 Model Storage Settings
S3_MODEL_BUCKET = os.getenv("S3_MODEL_BUCKET", "phdata-housing-models")
S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "models/housing")
# Specific model version to load (e.g., "20251202153000"). If empty, loads "latest"
S3_MODEL_VERSION = os.getenv("S3_MODEL_VERSION", "latest")
# Whether to load model from S3 (False = load from local MODEL_DIR)
USE_S3_MODEL = os.getenv("USE_S3_MODEL", "false").lower() == "true"
# Background model refresh interval in seconds (default: 600 = 10 minutes)
MODEL_REFRESH_INTERVAL_SECONDS = int(os.getenv("MODEL_REFRESH_INTERVAL_SECONDS", "600"))
# Enable automatic model refresh from S3
ENABLE_MODEL_AUTO_REFRESH = (
    os.getenv("ENABLE_MODEL_AUTO_REFRESH", "true").lower() == "true"
)

# JWT Settings (loaded from AWS Secrets Manager if USE_AWS_SECRETS=true)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError(
        "JWT_SECRET_KEY is required. Set it via environment variable or AWS Secrets Manager."
    )
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "30"))

# User credentials (loaded from AWS Secrets Manager if USE_AWS_SECRETS=true)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not ADMIN_USERNAME or not ADMIN_PASSWORD:
    raise ValueError(
        "ADMIN_USERNAME and ADMIN_PASSWORD are required. Set them via environment variables or AWS Secrets Manager."
    )
