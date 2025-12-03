"""
Pydantic Models for API Request/Response

This module defines the data models for the Housing Price Prediction API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime, timezone


class HouseFeatures(BaseModel):
    """
    Input model for house features.

    These are the features that a user provides to get a price prediction.
    Based on the columns in future_unseen_examples.csv (excluding price, date, id).
    """

    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(
        ..., ge=100, le=50000, description="Living area in square feet"
    )
    sqft_lot: int = Field(
        ..., ge=100, le=2000000, description="Lot size in square feet"
    )
    floors: float = Field(..., ge=1, le=4, description="Number of floors")
    waterfront: int = Field(
        default=0, ge=0, le=1, description="Waterfront property (0 or 1)"
    )
    view: int = Field(default=0, ge=0, le=4, description="View rating (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Condition rating (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Grade rating (1-13)")
    sqft_above: int = Field(..., ge=0, le=50000, description="Square feet above ground")
    sqft_basement: int = Field(
        default=0, ge=0, le=10000, description="Square feet of basement"
    )
    yr_built: int = Field(..., ge=1800, le=2025, description="Year the house was built")
    yr_renovated: int = Field(
        default=0, ge=0, le=2025, description="Year renovated (0 if never)"
    )
    zipcode: str = Field(
        ...,
        min_length=5,
        max_length=10,
        description="Zip code for demographic data lookup",
    )
    lat: Optional[float] = Field(default=None, description="Latitude")
    long: Optional[float] = Field(default=None, description="Longitude")
    sqft_living15: Optional[int] = Field(
        default=None, description="Living area of nearest 15 neighbors"
    )
    sqft_lot15: Optional[int] = Field(
        default=None, description="Lot size of nearest 15 neighbors"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bedrooms": 3,
                    "bathrooms": 2.0,
                    "sqft_living": 1800,
                    "sqft_lot": 5000,
                    "floors": 2,
                    "waterfront": 0,
                    "view": 0,
                    "condition": 3,
                    "grade": 7,
                    "sqft_above": 1800,
                    "sqft_basement": 0,
                    "yr_built": 1990,
                    "yr_renovated": 0,
                    "zipcode": "98001",
                }
            ]
        }
    }


class MinimalHouseFeatures(BaseModel):
    """
    Minimal input model with only the features required by the model.

    This schema only includes the house features that the ML model actually uses,
    plus zipcode for demographic data lookup.
    """

    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(
        ..., ge=100, le=50000, description="Living area in square feet"
    )
    sqft_lot: int = Field(
        ..., ge=100, le=2000000, description="Lot size in square feet"
    )
    floors: float = Field(..., ge=1, le=4, description="Number of floors")
    sqft_above: int = Field(..., ge=0, le=50000, description="Square feet above ground")
    sqft_basement: int = Field(
        default=0, ge=0, le=10000, description="Square feet of basement"
    )
    zipcode: str = Field(
        ...,
        min_length=5,
        max_length=10,
        description="Zip code for demographic data lookup",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bedrooms": 3,
                    "bathrooms": 2.0,
                    "sqft_living": 1800,
                    "sqft_lot": 5000,
                    "floors": 2,
                    "sqft_above": 1800,
                    "sqft_basement": 0,
                    "zipcode": "98001",
                }
            ]
        }
    }


class DemographicData(BaseModel):
    """
    Demographic data automatically joined from zipcode.
    Based on zipcode_demographics.csv columns.
    """

    zipcode: str
    median_household_income: Optional[float] = Field(
        default=None, description="Median household income"
    )
    population: Optional[float] = Field(default=None, description="Population quantity")
    housing_value: Optional[float] = Field(default=None, description="Housing value")


class PredictionResponse(BaseModel):
    """
    Response model for price prediction.
    """

    predicted_price: float = Field(..., description="Predicted house price in USD")
    confidence_interval: Dict[str, float] = Field(
        default_factory=dict, description="Confidence interval for the prediction"
    )
    house_features: HouseFeatures = Field(..., description="Input house features")
    demographic_data: DemographicData = Field(
        ..., description="Joined demographic data"
    )
    prediction_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the prediction",
    )
    model_version: str = Field(default="1.0.0", description="Version of the model used")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_price": 425000.00,
                    "confidence_interval": {"lower": 361250.00, "upper": 488750.00},
                    "house_features": {
                        "bedrooms": 3,
                        "bathrooms": 2.0,
                        "sqft_living": 1800,
                        "sqft_lot": 5000,
                        "floors": 2,
                        "waterfront": 0,
                        "view": 0,
                        "condition": 3,
                        "grade": 7,
                        "sqft_above": 1800,
                        "sqft_basement": 0,
                        "yr_built": 1990,
                        "yr_renovated": 0,
                        "zipcode": "98001",
                    },
                    "demographic_data": {
                        "zipcode": "98001",
                        "median_household_income": 60534,
                        "population": 25495,
                        "housing_value": 168400,
                    },
                    "prediction_timestamp": "2025-12-02T12:00:00Z",
                    "model_version": "1.0.0",
                }
            ]
        }
    }


class MinimalPredictionResponse(BaseModel):
    """
    Response model for minimal price prediction.
    """

    predicted_price: float = Field(..., description="Predicted house price in USD")
    confidence_interval: Dict[str, float] = Field(
        default_factory=dict, description="Confidence interval for the prediction"
    )
    house_features: MinimalHouseFeatures = Field(
        ..., description="Input house features"
    )
    demographic_data: DemographicData = Field(
        ..., description="Joined demographic data"
    )
    prediction_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the prediction",
    )
    model_version: str = Field(default="1.0.0", description="Version of the model used")


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch predictions.
    """

    houses: List[HouseFeatures] = Field(
        ..., description="List of houses to predict prices for"
    )


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions.
    """

    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    total_count: int = Field(..., description="Total number of predictions")
    prediction_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    """
    Error response model.
    """

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
