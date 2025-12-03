"""
API Routes

This module defines the REST API endpoints for the Housing Price Prediction API.
"""
from fastapi import APIRouter, HTTPException, status, Depends
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
from app.services.model_watcher import model_watcher
from app.auth import (
    authenticate_user,
    create_access_token,
    require_auth,
    Token,
    User,
)
from fastapi.security import OAuth2PasswordRequestForm
from config import API_VERSION


router = APIRouter()


@router.post("/login", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login to get a JWT access token.
    
    Use the returned token in the Authorization header as: Bearer <token>
    
    Default credentials (change via environment variables):
    - Username: admin (ADMIN_USERNAME)
    - Password: admin (ADMIN_PASSWORD)
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return Token(access_token=access_token, token_type="bearer")


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
        401: {"model": ErrorResponse, "description": "Unauthorized - Missing or invalid token"},
        503: {"model": ErrorResponse, "description": "Model not available"}
    },
    tags=["Predictions"]
)
async def predict_price(house: HouseFeatures, user: User = Depends(require_auth)):
    """
    Predict the price of a house.
    
    This endpoint receives JSON POST data with house details matching the columns
    in `data/future_unseen_examples.csv`. Demographic data is automatically joined
    from `data/zipcode_demographics.csv` based on the zipcode.
    
    **Input Features (from future_unseen_examples.csv):**
    - bedrooms: Number of bedrooms
    - bathrooms: Number of bathrooms
    - sqft_living: Living area in square feet
    - sqft_lot: Lot size in square feet
    - floors: Number of floors
    - waterfront: Waterfront property (0 or 1)
    - view: View rating (0-4)
    - condition: Condition rating (1-5)
    - grade: Grade rating (1-13)
    - sqft_above: Square feet above ground
    - sqft_basement: Square feet of basement
    - yr_built: Year the house was built
    - yr_renovated: Year renovated (0 if never)
    - zipcode: Zip code (for demographic data lookup)
    - lat: Latitude (optional)
    - long: Longitude (optional)
    - sqft_living15: Living area of nearest 15 neighbors (optional)
    - sqft_lot15: Lot size of nearest 15 neighbors (optional)
    
    **Returns:**
    - predicted_price: Estimated price in USD
    - confidence_interval: Price range estimate
    - house_features: Echo of input features
    - demographic_data: Joined demographic information from zipcode
    - prediction_timestamp: When the prediction was made
    - model_version: Version of the model used
    """
    if not prediction_service.is_model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please run 'python create_model.py' first."
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
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized - Missing or invalid token"},
        503: {"model": ErrorResponse, "description": "Model not available"}
    },
    tags=["Predictions"]
)
async def predict_batch(request: BatchPredictionRequest, user: User = Depends(require_auth)):
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


@router.get(
    "/model/info",
    response_model=ModelInfo,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized - Missing or invalid token"}
    },
    tags=["Model"]
)
async def get_model_info(user: User = Depends(require_auth)):
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


@router.post(
    "/model/reload",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized - Missing or invalid token"}
    },
    tags=["Model"]
)
async def reload_model(user: User = Depends(require_auth)):
    """
    Reload the model from disk.
    
    Use this endpoint after retraining the model to load the new version.
    """
    try:
        prediction_service.reload_model()
        return {"message": "Model and data reloaded successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )


@router.get(
    "/demographics/{zipcode}",
    response_model=DemographicData,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized - Missing or invalid token"}
    },
    tags=["Demographics"]
)
async def get_demographics(zipcode: str, user: User = Depends(require_auth)):
    """
    Get demographic data for a specific zip code.
    
    This endpoint returns the demographic information that would be
    automatically joined when making a prediction.
    """
    demographics = prediction_service._get_demographics_for_zipcode(zipcode)
    return DemographicData(
        zipcode=zipcode,
        median_household_income=demographics.get('medn_hshld_incm_amt'),
        population=demographics.get('ppltn_qty'),
        housing_value=demographics.get('hous_val_amt'),
    )


@router.get(
    "/model/watcher/status",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized - Missing or invalid token"}
    },
    tags=["Model Management"]
)
async def get_model_watcher_status(user: User = Depends(require_auth)):
    """
    Get the status of the background model watcher.
    
    Returns information about:
    - Whether auto-refresh is enabled
    - Refresh interval
    - Last check timestamp
    - Total checks and reloads performed
    - Current model version
    """
    return model_watcher.get_status()


@router.post(
    "/model/watcher/check",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized - Missing or invalid token"}
    },
    tags=["Model Management"]
)
async def trigger_model_check(user: User = Depends(require_auth)):
    """
    Manually trigger a model version check.
    
    This endpoint forces an immediate check for new model versions in S3,
    bypassing the regular interval. If a new version is found, the model
    will be hot-reloaded.
    """
    result = await model_watcher.check_and_reload()
    return result
