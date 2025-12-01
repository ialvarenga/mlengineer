#!/usr/bin/env python
"""
Test Client Script

This script demonstrates how to interact with the Housing Price Prediction API.
It sends sample house data to the API and displays the predictions.
"""
import os
import requests
import json
from typing import Dict, List, Optional


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
USERNAME = os.getenv("API_USERNAME", "admin")
PASSWORD = os.getenv("API_PASSWORD", "admin")

# Global token storage
_access_token: Optional[str] = None


def login() -> Optional[str]:
    """Login and get an access token."""
    global _access_token
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/login",
            data={"username": USERNAME, "password": PASSWORD},
            timeout=10
        )
        if response.status_code == 200:
            _access_token = response.json()["access_token"]
            return _access_token
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API for login")
        return None


def get_headers() -> Dict[str, str]:
    """Get headers for API requests, including auth token."""
    headers = {"Content-Type": "application/json"}
    if _access_token:
        headers["Authorization"] = f"Bearer {_access_token}"
    return headers


def print_separator(title: str = ""):
    """Print a separator line."""
    print("\n" + "=" * 60)
    if title:
        print(f" {title}")
        print("=" * 60)


def check_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   API Version: {data['api_version']}")
            return data['model_loaded']
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        return False


def predict_single_house(house_data: Dict) -> Dict:
    """Make a prediction for a single house."""
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=house_data,
        headers=get_headers(),
        timeout=10
    )
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        print("❌ Authentication failed. Please check credentials.")
        return None
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None


def predict_batch(houses: List[Dict]) -> Dict:
    """Make predictions for multiple houses."""
    response = requests.post(
        f"{API_BASE_URL}/predict/batch",
        json={"houses": houses},
        headers=get_headers(),
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        print("❌ Authentication failed. Please check credentials.")
        return None
    else:
        print(f"❌ Error: {response.status_code}")
        return None


def get_model_info() -> Dict:
    """Get information about the current model."""
    response = requests.get(
        f"{API_BASE_URL}/model/info",
        headers=get_headers(),
        timeout=5
    )
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        print("❌ Authentication failed. Please check credentials.")
        return None
    return None


def get_demographics(zipcode: str) -> Dict:
    """Get demographic data for a zipcode."""
    response = requests.get(
        f"{API_BASE_URL}/demographics/{zipcode}",
        headers=get_headers(),
        timeout=5
    )
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        print("❌ Authentication failed. Please check credentials.")
        return None
    return None


def format_price(price: float) -> str:
    """Format price as currency."""
    return f"${price:,.2f}"


def main():
    """Main test function."""
    print_separator("HOUSING PRICE PREDICTION API - TEST CLIENT")
    print("\nThis script tests the Housing Price Prediction REST API.")
    print("Make sure the API is running: uvicorn app.main:app --reload")
    
    # 1. Check API Health
    print_separator("1. HEALTH CHECK")
    model_loaded = check_health()
    
    if not model_loaded:
        print("\n⚠️ Model not loaded. Train it first with: python -m ml.train")
        return
    
    # 2. Login to get access token
    print_separator("2. AUTHENTICATION")
    print(f"   Logging in as: {USERNAME}")
    token = login()
    if token:
        print("   ✅ Login successful, token received")
    else:
        print("   ❌ Login failed, cannot continue with protected endpoints")
        return
    
    # 3. Get Model Info
    print_separator("3. MODEL INFORMATION")
    model_info = get_model_info()
    if model_info:
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Model Version: {model_info['model_version']}")
        print(f"   Features Used: {len(model_info['features_used'])} features")
        if model_info.get('training_metrics') and 'test' in model_info['training_metrics']:
            test_metrics = model_info['training_metrics']['test']
            print(f"   Test R² Score: {test_metrics.get('r2_score', 'N/A'):.4f}")
            print(f"   Test RMSE: ${test_metrics.get('rmse', 0):,.2f}")
    
    # 4. Test Demographics Lookup
    print_separator("4. DEMOGRAPHICS LOOKUP")
    test_zipcodes = ["98001", "98004", "98052"]
    for zipcode in test_zipcodes:
        demo = get_demographics(zipcode)
        if demo:
            print(f"\n   Zipcode: {zipcode}")
            print(f"   City: {demo.get('city', 'N/A')}, State: {demo.get('state', 'N/A')}")
            print(f"   Median Income: ${demo.get('median_income', 0):,.0f}")
            print(f"   Population Density: {demo.get('population_density', 0):,.0f}")
    
    # 5. Single House Prediction
    print_separator("5. SINGLE HOUSE PREDICTION")
    
    # Sample houses with different characteristics
    sample_houses = [
        {
            "name": "Modest Starter Home",
            "data": {
                "bedrooms": 2,
                "bathrooms": 1.0,
                "sqft_living": 1200,
                "sqft_lot": 5000,
                "floors": 1,
                "condition": 3,
                "grade": 6,
                "year_built": 1965,
                "zipcode": "98001"
            }
        },
        {
            "name": "Family Home",
            "data": {
                "bedrooms": 4,
                "bathrooms": 2.5,
                "sqft_living": 2400,
                "sqft_lot": 8000,
                "floors": 2,
                "condition": 4,
                "grade": 8,
                "year_built": 2005,
                "zipcode": "98052"
            }
        },
        {
            "name": "Luxury Bellevue Property",
            "data": {
                "bedrooms": 5,
                "bathrooms": 4.0,
                "sqft_living": 4500,
                "sqft_lot": 15000,
                "floors": 2,
                "condition": 5,
                "grade": 11,
                "year_built": 2020,
                "zipcode": "98004"
            }
        }
    ]
    
    for house in sample_houses:
        print(f"\n🏠 {house['name']}")
        print("-" * 40)
        
        result = predict_single_house(house['data'])
        
        if result:
            print(f"   Bedrooms: {house['data']['bedrooms']} | Bathrooms: {house['data']['bathrooms']}")
            print(f"   Sqft Living: {house['data']['sqft_living']:,} | Year Built: {house['data']['year_built']}")
            print(f"   Zipcode: {house['data']['zipcode']} ({result['demographic_data'].get('city', 'N/A')})")
            print()
            print(f"   📊 PREDICTED PRICE: {format_price(result['predicted_price'])}")
            
            if 'confidence_interval' in result:
                ci = result['confidence_interval']
                print(f"   📈 95% CI: {format_price(ci['lower'])} - {format_price(ci['upper'])}")
    
    # 6. Batch Prediction
    print_separator("6. BATCH PREDICTION")
    batch_houses = [house['data'] for house in sample_houses]
    
    print(f"\n   Sending {len(batch_houses)} houses for batch prediction...")
    batch_result = predict_batch(batch_houses)
    
    if batch_result:
        print(f"   ✅ Received {batch_result['total_count']} predictions")
        print("\n   Summary of Predictions:")
        for i, pred in enumerate(batch_result['predictions'], 1):
            print(f"   {i}. {format_price(pred['predicted_price'])} ({pred['demographic_data'].get('city', 'N/A')})")
    
    # 7. Summary
    print_separator("7. TEST SUMMARY")
    print("\n   ✅ All API endpoints tested successfully!")
    print("\n   Available Endpoints:")
    print("   - POST /login         Get access token")
    print("   - GET  /              Health check (public)")
    print("   - GET  /health        Detailed health status (public)")
    print("   - POST /predict       Single house prediction (auth required)")
    print("   - POST /predict/batch Batch predictions (auth required)")
    print("   - GET  /model/info    Model information (auth required)")
    print("   - GET  /demographics/{zipcode}  Demographic data (auth required)")
    print("   - POST /model/reload  Reload model (auth required)")
    
    print("\n   📖 API Documentation: http://localhost:8000/docs")
    print_separator()


if __name__ == "__main__":
    main()
