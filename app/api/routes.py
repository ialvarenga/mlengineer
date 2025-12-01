"""
API Routes

This module defines the REST API endpoints for the Housing Price Prediction API.
"""
from fastapi import APIRouter, HTTPException, status
from typing import List
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schemas import (
    HouseFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
    DemographicData
)
from app.services.prediction import prediction_service
from app.services.demographics import demographics_service
from config import API_VERSION


router = APIRouter()


@router.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """
    Root endpoint - basic health check.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service.is_model_loaded,
        api_version=API_VERSION
    )


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Detailed health check endpoint.
    
    Returns the current status of the API and model.
    """
    return HealthResponse(
        status="healthy" if prediction_service.is_model_loaded else "degraded",
        model_loaded=prediction_service.is_model_loaded,
        api_version=API_VERSION
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not available"}
    },
    tags=["Predictions"]
)
async def predict_price(house: HouseFeatures):
    """
    Predict the price of a house.
    
    This endpoint takes house details as input and automatically joins
    zip code demographic data behind the scenes to make a prediction.
    
    **Input Features:**
    - bedrooms: Number of bedrooms
    - bathrooms: Number of bathrooms
    - sqft_living: Living area in square feet
    - sqft_lot: Lot size in square feet
    - floors: Number of floors
    - condition: Condition rating (1-5)
    - grade: Grade rating (1-13)
    - year_built: Year the house was built
    - zipcode: Zip code (for demographic data lookup)
    
    **Returns:**
    - predicted_price: Estimated price in USD
    - confidence_interval: Price range with 95% confidence
    - demographic_data: Joined demographic information
    """
    if not prediction_service.is_model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first by running: python -m ml.train"
        )
    
    try:
        # Convert Pydantic model to dict
        house_dict = house.model_dump()
        
        # Get prediction
        result = prediction_service.predict(house_dict)
        
        # Build response
        return PredictionResponse(
            predicted_price=result['predicted_price'],
            confidence_interval=result['confidence_interval'],
            house_features=house,
            demographic_data=DemographicData(**result['demographic_data']),
            model_version=result['model_version']
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"]
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict prices for multiple houses at once.
    
    This endpoint accepts a list of houses and returns predictions for all of them.
    """
    if not prediction_service.is_model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    predictions = []
    for house in request.houses:
        try:
            house_dict = house.model_dump()
            result = prediction_service.predict(house_dict)
            
            predictions.append(PredictionResponse(
                predicted_price=result['predicted_price'],
                confidence_interval=result['confidence_interval'],
                house_features=house,
                demographic_data=DemographicData(**result['demographic_data']),
                model_version=result['model_version']
            ))
        except Exception as e:
            logger.error(f"Batch prediction error for house: {e}")
            # Continue with other predictions
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_count=len(predictions)
    )


@router.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get information about the current model.
    
    Returns details about:
    - Model type
    - Model version
    - Features used
    - Training metrics
    - Feature importance
    """
    info = prediction_service.get_model_info()
    return ModelInfo(
        model_type=info['model_type'],
        model_version=info['model_version'],
        features_used=info['features_used'],
        training_metrics=info.get('training_metrics', {})
    )


@router.post("/model/reload", tags=["Model"])
async def reload_model():
    """
    Reload the model from disk.
    
    Use this endpoint after retraining the model to load the new version.
    """
    try:
        prediction_service.reload_model()
        demographics_service.reload()
        return {"message": "Model and data reloaded successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )


@router.get("/demographics/{zipcode}", tags=["Demographics"])
async def get_demographics(zipcode: str):
    """
    Get demographic data for a specific zip code.
    
    This endpoint returns the demographic information that would be
    automatically joined when making a prediction.
    """
    demographics = demographics_service.get_demographics(zipcode)
    return DemographicData(**demographics)
