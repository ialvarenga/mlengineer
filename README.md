# Housing Price Prediction API

A machine learning project that predicts housing prices using a REST API, developed as part of the PhData ML Engineer interview challenge.

## Project Overview

This project implements:
- A trained ML model for housing price prediction
- A REST API (FastAPI) that accepts house details and returns price estimates
- Automatic integration of zip code demographic data
- Model evaluation and performance metrics
- A test client script to demonstrate API functionality

## Project Structure

```
mlengineer/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── prediction.py    # Prediction service
│       └── demographics.py  # Zip code data service
├── ml/
│   ├── __init__.py
│   ├── model.py             # ML model definition
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Model evaluation
│   └── artifacts/           # Saved models
├── data/
│   ├── housing_data.csv     # Training data
│   └── zipcode_demographics.csv  # Demographic data
├── tests/
│   ├── __init__.py
│   └── test_api.py          # API tests
├── scripts/
│   └── test_client.py       # Demo client script
├── notebooks/
│   └── model_evaluation.ipynb  # Evaluation notebook
├── docker/
│   └── Dockerfile           # Container config
├── requirements.txt
├── config.py
└── README.md
```

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python -m ml.train
```

### 4. Run the API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test the API
```bash
python scripts/test_client.py
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - API health status
- `POST /predict` - Get price prediction for a house
- `GET /model/info` - Get model information and metrics

## Example Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 2,
        "condition": 3,
        "grade": 7,
        "year_built": 1990,
        "zipcode": "98001"
    }
)
print(response.json())
```

## Model Evaluation

The model evaluation includes:
- R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Overfitting/Underfitting analysis
- Feature importance analysis

## MLOps Considerations

- **Model Registry**: Version control for models
- **Feature Store**: Centralized feature management
- **CI/CD**: Automated testing and deployment
- **Monitoring**: Model performance tracking
- **Versioning**: API and model versioning

## AWS Deployment Considerations

- **ECS/EKS**: Container orchestration
- **ECR**: Container registry
- **SageMaker**: Model hosting and endpoints
- **CloudWatch**: Monitoring and logging
- **API Gateway**: API management

## Author

Developed for PhData ML Engineer Interview Challenge
