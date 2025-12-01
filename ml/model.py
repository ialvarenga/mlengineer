"""
Housing Price Prediction Model

This module defines the ML model for predicting housing prices.
The model uses a Random Forest Regressor with feature engineering
and proper preprocessing pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_PATH, SCALER_PATH, RANDOM_STATE


class HousingPriceModel:
    """
    Housing Price Prediction Model
    
    This model predicts housing prices based on property features
    and demographic data from the property's zip code.
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'ridge')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        self._create_model()
    
    def _create_model(self):
        """Create the underlying ML model based on model_type."""
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE
            )
        elif self.model_type == "ridge":
            self.model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Created {self.model_type} model")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'HousingPriceModel':
        """
        Train the model on the provided data.
        
        Args:
            X: Feature DataFrame
            y: Target Series (prices)
            
        Returns:
            self
        """
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Model trained on {len(X)} samples with {len(self.feature_names)} features")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted prices
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Ensure feature order matches training
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self) -> dict:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            # For models without feature_importances_, return equal weights
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)
        
        return dict(zip(self.feature_names, importances))
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
        """
        Perform cross-validation and return scores.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with CV scores
        """
        X_scaled = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='r2')
        
        return {
            'cv_scores': scores.tolist(),
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std()
        }
    
    def save(self, model_path: Path = None, scaler_path: Path = None):
        """
        Save the model and scaler to disk.
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler
        """
        model_path = model_path or MODEL_PATH
        scaler_path = scaler_path or SCALER_PATH
        
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
    
    @classmethod
    def load(cls, model_path: Path = None, scaler_path: Path = None) -> 'HousingPriceModel':
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler
            
        Returns:
            Loaded HousingPriceModel instance
        """
        model_path = model_path or MODEL_PATH
        scaler_path = scaler_path or SCALER_PATH
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model data
        model_data = joblib.load(model_path)
        
        # Create instance
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        # Load scaler
        if scaler_path.exists():
            instance.scaler = joblib.load(scaler_path)
        
        logger.info(f"Model loaded from {model_path}")
        
        return instance
