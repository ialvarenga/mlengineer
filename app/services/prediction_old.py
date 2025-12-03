"""
Prediction Service

This service handles the ML model loading and prediction logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from loguru import logger

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import MODEL_PATH, SCALER_PATH, MODEL_VERSION, MODEL_METRICS_PATH
from ml.model import HousingPriceModel
from app.services.demographics import demographics_service


class PredictionService:
    """
    Service for making housing price predictions.
    """

    _instance = None
    _model: Optional[HousingPriceModel] = None
    _model_metrics: Optional[Dict] = None

    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """Load the ML model from disk."""
        try:
            self._model = HousingPriceModel.load(MODEL_PATH, SCALER_PATH)
            self._load_metrics()
            logger.info("ML model loaded successfully")
        except FileNotFoundError:
            logger.warning("Model file not found. Please train the model first.")
            self._model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model = None

    def _load_metrics(self):
        """Load model metrics from file."""
        import json

        try:
            if MODEL_METRICS_PATH.exists():
                with open(MODEL_METRICS_PATH, "r") as f:
                    self._model_metrics = json.load(f)
                logger.info("Model metrics loaded")
        except Exception as e:
            logger.warning(f"Could not load metrics: {e}")
            self._model_metrics = {}

    @property
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def predict(self, house_features: Dict) -> Dict:
        """
        Make a price prediction for a house.

        Args:
            house_features: Dictionary with house features including zipcode

        Returns:
            Dictionary with prediction results
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded. Please train the model first.")

        # Extract zipcode and get demographic data
        zipcode = str(house_features.get("zipcode", ""))
        demographics = demographics_service.get_demographics(zipcode)
        demo_features = demographics_service.get_model_features(zipcode)

        # Prepare features for the model
        model_features = {
            "bedrooms": house_features["bedrooms"],
            "bathrooms": house_features["bathrooms"],
            "sqft_living": house_features["sqft_living"],
            "sqft_lot": house_features["sqft_lot"],
            "floors": house_features["floors"],
            "condition": house_features["condition"],
            "grade": house_features["grade"],
            "year_built": house_features["year_built"],
            **demo_features,  # Add demographic features
        }

        # Create DataFrame for prediction
        X = pd.DataFrame([model_features])

        # Make prediction
        predicted_price = self._model.predict(X)[0]

        # Calculate confidence interval (simple estimation based on model error)
        # In production, this would use prediction intervals from the model
        rmse = self._get_model_rmse()
        lower_bound = max(0, predicted_price - 1.96 * rmse)
        upper_bound = predicted_price + 1.96 * rmse

        return {
            "predicted_price": round(float(predicted_price), 2),
            "confidence_interval": {
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2),
            },
            "demographic_data": demographics,
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "model_version": MODEL_VERSION,
        }

    def _get_model_rmse(self) -> float:
        """Get the model's RMSE for confidence interval calculation."""
        if self._model_metrics and "test" in self._model_metrics:
            return self._model_metrics["test"].get("rmse", 50000)
        return 50000  # Default RMSE if not available

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        if not self.is_model_loaded:
            return {
                "model_type": "Not loaded",
                "model_version": MODEL_VERSION,
                "features_used": [],
                "training_metrics": {},
                "status": "Model not loaded",
            }

        return {
            "model_type": self._model.model_type,
            "model_version": MODEL_VERSION,
            "features_used": self._model.feature_names or [],
            "training_metrics": self._model_metrics or {},
            "feature_importance": (
                self._model.get_feature_importance() if self._model.is_fitted else {}
            ),
        }

    def reload_model(self):
        """Reload the model from disk."""
        self._load_model()
        logger.info("Model reloaded")


# Global instance
prediction_service = PredictionService()
