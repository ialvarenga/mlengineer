"""
Housing Price Prediction API

Main FastAPI application module.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from contextlib import asynccontextmanager
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import API_TITLE, API_DESCRIPTION, API_VERSION
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    """
    # Startup
    logger.info("Starting Housing Price Prediction API...")
    logger.info(f"API Version: {API_VERSION}")
    
    # Check if model is loaded
    from app.services.prediction import prediction_service
    if prediction_service.is_model_loaded:
        logger.info("✅ ML Model loaded successfully")
    else:
        logger.warning("⚠️ ML Model not loaded. Train the model first: python -m ml.train")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/openapi-info")
async def openapi_info():
    """
    Get basic API information.
    """
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
