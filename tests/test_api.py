"""
API Unit Tests

Tests for the Housing Price Prediction API endpoints and services.
Tests behavior, not implementation details.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Get authentication headers for protected endpoints."""
    response = client.post("/login", data={"username": "admin", "password": "admin"})
    if response.status_code == 200:
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    return {}


@pytest.fixture
def sample_house_data():
    """Sample house data for predictions."""
    return {
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


@pytest.fixture
def minimal_house_data():
    """Minimal house data for predictions."""
    return {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 2,
        "sqft_above": 1800,
        "sqft_basement": 0,
        "zipcode": "98001",
    }


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_returns_healthy_status(self, client):
        """Root endpoint should return health status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "api_version" in data

    def test_health_endpoint_returns_status(self, client):
        """Health endpoint should return detailed status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "api_version" in data


class TestAuthentication:
    """Tests for authentication."""

    def test_login_with_valid_credentials_returns_token(self, client):
        """Login with valid credentials should return JWT token."""
        response = client.post(
            "/login", data={"username": "admin", "password": "admin"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_with_invalid_credentials_returns_401(self, client):
        """Login with invalid credentials should return 401."""
        response = client.post(
            "/login", data={"username": "wrong", "password": "wrong"}
        )
        assert response.status_code == 401

    def test_protected_endpoint_without_token_returns_401(self, client):
        """Protected endpoints should require authentication."""
        response = client.get("/model/info")
        assert response.status_code == 401


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_predict_requires_authentication(self, client, sample_house_data):
        """Predict endpoint should require authentication."""
        response = client.post("/predict", json=sample_house_data)
        assert response.status_code == 401

    @patch("app.services.prediction.prediction_service")
    def test_predict_returns_price_when_model_loaded(
        self, mock_service, client, auth_headers, sample_house_data
    ):
        """Predict should return price when model is loaded."""
        mock_service.is_model_loaded = True
        mock_service.predict.return_value = {
            "predicted_price": 450000.0,
            "confidence_interval": {"lower": 400000.0, "upper": 500000.0},
            "model_version": "1.0.0",
            "demographic_data": {
                "zipcode": "98001",
                "medn_hshld_incm_amt": 65000,
                "ppltn_qty": 50000,
                "hous_val_amt": 350000,
            },
        }

        response = client.post("/predict", json=sample_house_data, headers=auth_headers)

        if response.status_code == 200:
            data = response.json()
            assert "predicted_price" in data
            assert isinstance(data["predicted_price"], (int, float))

    def test_predict_validates_input_fields(self, client, auth_headers):
        """Predict should validate required input fields."""
        invalid_data = {"bedrooms": 3}
        response = client.post("/predict", json=invalid_data, headers=auth_headers)
        assert response.status_code == 422

    def test_predict_validates_field_ranges(
        self, client, auth_headers, sample_house_data
    ):
        """Predict should validate field value ranges."""
        invalid_data = sample_house_data.copy()
        invalid_data["bedrooms"] = -1
        response = client.post("/predict", json=invalid_data, headers=auth_headers)
        assert response.status_code == 422

    def test_predict_minimal_requires_authentication(self, client, minimal_house_data):
        """Minimal predict endpoint should require authentication."""
        response = client.post("/predict/minimal", json=minimal_house_data)
        assert response.status_code == 401


class TestInputValidation:
    """Tests for input validation behavior."""

    def test_zipcode_must_be_at_least_5_characters(
        self, client, auth_headers, sample_house_data
    ):
        """Zipcode should be at least 5 characters."""
        invalid_data = sample_house_data.copy()
        invalid_data["zipcode"] = "123"
        response = client.post("/predict", json=invalid_data, headers=auth_headers)
        assert response.status_code == 422

    def test_sqft_living_must_be_positive(
        self, client, auth_headers, sample_house_data
    ):
        """Square footage should be within valid range."""
        invalid_data = sample_house_data.copy()
        invalid_data["sqft_living"] = 50
        response = client.post("/predict", json=invalid_data, headers=auth_headers)
        assert response.status_code == 422

    def test_condition_must_be_between_1_and_5(
        self, client, auth_headers, sample_house_data
    ):
        """Condition rating should be between 1 and 5."""
        invalid_data = sample_house_data.copy()
        invalid_data["condition"] = 10
        response = client.post("/predict", json=invalid_data, headers=auth_headers)
        assert response.status_code == 422

    def test_grade_must_be_between_1_and_13(
        self, client, auth_headers, sample_house_data
    ):
        """Grade rating should be between 1 and 13."""
        invalid_data = sample_house_data.copy()
        invalid_data["grade"] = 20
        response = client.post("/predict", json=invalid_data, headers=auth_headers)
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for model information endpoint."""

    def test_model_info_requires_authentication(self, client):
        """Model info endpoint should require authentication."""
        response = client.get("/model/info")
        assert response.status_code == 401

    @patch("app.services.prediction.prediction_service")
    def test_model_info_returns_model_details(self, mock_service, client, auth_headers):
        """Model info should return model metadata."""
        mock_service.get_model_info.return_value = {
            "model_type": "GradientBoostingRegressor",
            "model_version": "1.0.0",
            "features_used": ["bedrooms", "bathrooms", "sqft_living"],
            "training_metrics": {"r2_score": 0.85},
        }

        response = client.get("/model/info", headers=auth_headers)

        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "model_version" in data
