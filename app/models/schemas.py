"""
Pydantic Models for API Request/Response

This module defines the data models for the Housing Price Prediction API.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


class HouseFeatures(BaseModel):
    """
    Input model for house features.
    
    These are the features that a user provides to get a price prediction.
    """
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=100, le=50000, description="Living area in square feet")
    sqft_lot: int = Field(..., ge=100, le=1000000, description="Lot size in square feet")
    floors: float = Field(..., ge=1, le=4, description="Number of floors")
    condition: int = Field(..., ge=1, le=5, description="Condition rating (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Grade rating (1-13)")
    year_built: int = Field(..., ge=1800, le=2025, description="Year the house was built")
    zipcode: str = Field(..., min_length=5, max_length=10, description="Zip code for demographic data lookup")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bedrooms": 3,
                    "bathrooms": 2.0,
                    "sqft_living": 1800,
                    "sqft_lot": 5000,
                    "floors": 2,
                    "condition": 3,
                    "grade": 7,
                    "year_built": 1990,
                    "zipcode": "98001"
                }
            ]
        }
    }


class DemographicData(BaseModel):
    """
    Demographic data automatically joined from zipcode.
    """
    zipcode: str
    median_income: Optional[float] = None
    population_density: Optional[float] = None
    median_age: Optional[float] = None
    city: Optional[str] = None
    state: Optional[str] = None


class PredictionResponse(BaseModel):
    """
    Response model for price prediction.
    """
    predicted_price: float = Field(..., description="Predicted house price in USD")
    confidence_interval: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence interval for the prediction"
    )
    house_features: HouseFeatures = Field(..., description="Input house features")
    demographic_data: DemographicData = Field(..., description="Joined demographic data")
    prediction_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the prediction"
    )
    model_version: str = Field(default="1.0.0", description="Version of the model used")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_price": 425000.00,
                    "confidence_interval": {
                        "lower": 380000.00,
                        "upper": 470000.00
                    },
                    "house_features": {
                        "bedrooms": 3,
                        "bathrooms": 2.0,
                        "sqft_living": 1800,
                        "sqft_lot": 5000,
                        "floors": 2,
                        "condition": 3,
                        "grade": 7,
                        "year_built": 1990,
                        "zipcode": "98001"
                    },
                    "demographic_data": {
                        "zipcode": "98001",
                        "median_income": 58000,
                        "population_density": 2850,
                        "median_age": 34.5,
                        "city": "Auburn",
                        "state": "WA"
                    },
                    "prediction_timestamp": "2025-11-30T12:00:00Z",
                    "model_version": "1.0.0"
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch predictions.
    """
    houses: List[HouseFeatures] = Field(..., description="List of houses to predict prices for")


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions.
    """
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfo(BaseModel):
    """
    Information about the current model.
    """
    model_type: str = Field(..., description="Type of ML model")
    model_version: str = Field(..., description="Model version")
    features_used: List[str] = Field(..., description="Features used by the model")
    training_metrics: Dict = Field(..., description="Model training metrics")
    last_updated: Optional[datetime] = None


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    api_version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """
    Error response model.
    """
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
