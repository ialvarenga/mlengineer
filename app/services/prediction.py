"""
Prediction Service

This service handles the ML model loading and prediction logic.
Uses the model created by create_model.py (Sound Realty's basic model).
"""
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import MODEL_DIR, MODEL_PATH, MODEL_FEATURES_PATH, ZIPCODE_DATA_PATH, MODEL_VERSION


class PredictionService:
    """
    Service for making housing price predictions using the Sound Realty model.
    """
    
    _instance = None
    _model = None
    _model_features: List[str] = []
    _demographics_df: Optional[pd.DataFrame] = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
            cls._instance._load_demographics()
        return cls._instance
    
    def _load_model(self):
        """Load the ML model and features from disk."""
        try:
            # Load the pickled model (sklearn pipeline)
            with open(MODEL_PATH, 'rb') as f:
                self._model = pickle.load(f)
            
            # Load the feature names
            with open(MODEL_FEATURES_PATH, 'r') as f:
                self._model_features = json.load(f)
            
            logger.info(f"ML model loaded successfully with {len(self._model_features)} features")
        except FileNotFoundError as e:
            logger.warning(f"Model file not found: {e}. Please run 'python create_model.py' first.")
            self._model = None
            self._model_features = []
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model = None
            self._model_features = []
    
    def _load_demographics(self):
        """Load demographic data from CSV file."""
        try:
            self._demographics_df = pd.read_csv(ZIPCODE_DATA_PATH, dtype={'zipcode': str})
            logger.info(f"Loaded demographics data for {len(self._demographics_df)} zip codes")
        except FileNotFoundError:
            logger.warning(f"Demographics file not found: {ZIPCODE_DATA_PATH}")
            self._demographics_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading demographics: {e}")
            self._demographics_df = pd.DataFrame()
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    def _get_demographics_for_zipcode(self, zipcode: str) -> Dict:
        """Get demographic data for a specific zipcode."""
        if self._demographics_df is None or self._demographics_df.empty:
            return {}
        
        match = self._demographics_df[self._demographics_df['zipcode'] == str(zipcode)]
        if match.empty:
            # Return median values for unknown zipcodes
            logger.warning(f"Zipcode {zipcode} not found, using median values")
            return self._demographics_df.drop(columns=['zipcode']).median().to_dict()
        
        return match.iloc[0].drop('zipcode').to_dict()
    
    def predict(self, house_features: Dict) -> Dict:
        """
        Make a price prediction for a house.
        
        Args:
            house_features: Dictionary with house features including zipcode
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded. Please run 'python create_model.py' first.")
        
        # Extract zipcode and get demographic data
        zipcode = str(house_features.get('zipcode', ''))
        demographics = self._get_demographics_for_zipcode(zipcode)
        
        # Build the feature dict matching the model's expected features
        # Start with house features (from the sales data columns used in create_model.py)
        feature_dict = {
            'bedrooms': house_features.get('bedrooms'),
            'bathrooms': house_features.get('bathrooms'),
            'sqft_living': house_features.get('sqft_living'),
            'sqft_lot': house_features.get('sqft_lot'),
            'floors': house_features.get('floors'),
            'sqft_above': house_features.get('sqft_above', house_features.get('sqft_living', 0)),
            'sqft_basement': house_features.get('sqft_basement', 0),
        }
        
        # Add demographic features
        feature_dict.update(demographics)
        
        # Create DataFrame with features in the correct order
        X = pd.DataFrame([feature_dict])
        
        # Ensure all model features are present, fill missing with 0
        for feat in self._model_features:
            if feat not in X.columns:
                X[feat] = 0
                logger.warning(f"Feature '{feat}' not found, using 0")
        
        # Reorder columns to match model training order
        X = X[self._model_features]
        
        # Make prediction
        predicted_price = self._model.predict(X)[0]
        
        # Simple confidence interval estimation (±15% as rough estimate)
        margin = predicted_price * 0.15
        lower_bound = max(0, predicted_price - margin)
        upper_bound = predicted_price + margin
        
        return {
            'predicted_price': round(float(predicted_price), 2),
            'confidence_interval': {
                'lower': round(lower_bound, 2),
                'upper': round(upper_bound, 2)
            },
            'demographic_data': {
                'zipcode': zipcode,
                'median_household_income': demographics.get('medn_hshld_incm_amt', 0),
                'population': demographics.get('ppltn_qty', 0),
                'housing_value': demographics.get('hous_val_amt', 0),
            },
            'prediction_timestamp': datetime.utcnow().isoformat(),
            'model_version': MODEL_VERSION
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        if not self.is_model_loaded:
            return {
                'model_type': 'Not loaded',
                'model_version': MODEL_VERSION,
                'features_used': [],
                'status': 'Model not loaded. Run python create_model.py'
            }
        
        # Get model type from the pipeline
        model_type = "KNeighborsRegressor Pipeline"
        if hasattr(self._model, 'steps'):
            model_type = ' -> '.join([step[0] for step in self._model.steps])
        
        return {
            'model_type': model_type,
            'model_version': MODEL_VERSION,
            'features_used': self._model_features,
            'training_metrics': {},
        }
    
    def reload_model(self):
        """Reload the model from disk."""
        self._load_model()
        self._load_demographics()
        logger.info("Model and demographics reloaded")


# Global instance
prediction_service = PredictionService()
