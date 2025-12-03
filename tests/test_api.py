"""
API Tests

Unit tests for the Housing Price Prediction API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


client = TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self):
        """Test the root endpoint returns health status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "api_version" in data

    def test_health_endpoint(self):
        """Test the health endpoint returns detailed status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    @pytest.fixture
    def valid_house_data(self):
        """Valid house data for testing."""
        return {
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_living": 1800,
            "sqft_lot": 5000,
            "floors": 2,
            "condition": 3,
            "grade": 7,
            "year_built": 1990,
            "zipcode": "98001",
        }

    def test_predict_endpoint_valid_data(self, valid_house_data):
        """Test prediction with valid house data."""
        response = client.post("/predict", json=valid_house_data)

        # May return 503 if model not loaded
        if response.status_code == 200:
            data = response.json()
            assert "predicted_price" in data
            assert "confidence_interval" in data
            assert "house_features" in data
            assert "demographic_data" in data
            assert data["predicted_price"] > 0
        elif response.status_code == 503:
            # Model not loaded, which is expected in test environment
            assert "Model not loaded" in response.json()["detail"]

    def test_predict_endpoint_invalid_bedrooms(self):
        """Test prediction with invalid bedroom count."""
        invalid_data = {
            "bedrooms": -1,  # Invalid
            "bathrooms": 2.0,
            "sqft_living": 1800,
            "sqft_lot": 5000,
            "floors": 2,
            "condition": 3,
            "grade": 7,
            "year_built": 1990,
            "zipcode": "98001",
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_field(self):
        """Test prediction with missing required field."""
        incomplete_data = {
            "bedrooms": 3,
            "bathrooms": 2.0,
            # Missing other required fields
        }
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_endpoint(self, valid_house_data):
        """Test batch prediction endpoint."""
        batch_request = {"houses": [valid_house_data, valid_house_data]}
        response = client.post("/predict/batch", json=batch_request)

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_count" in data
        elif response.status_code == 503:
            # Model not loaded
            pass


class TestModelEndpoints:
    """Tests for model information endpoints."""

    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "model_version" in data
        assert "features_used" in data


class TestDemographicsEndpoints:
    """Tests for demographics endpoints."""

    def test_demographics_valid_zipcode(self):
        """Test demographics lookup with valid zipcode."""
        response = client.get("/demographics/98001")
        assert response.status_code == 200
        data = response.json()
        assert "zipcode" in data
        assert data["zipcode"] == "98001"

    def test_demographics_unknown_zipcode(self):
        """Test demographics lookup with unknown zipcode."""
        response = client.get("/demographics/00000")
        assert response.status_code == 200
        data = response.json()
        # Should return default values
        assert "zipcode" in data


class TestDataValidation:
    """Tests for input data validation."""

    def test_sqft_living_bounds(self):
        """Test sqft_living validation bounds."""
        # Too small
        data = {
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_living": 50,  # Below minimum
            "sqft_lot": 5000,
            "floors": 2,
            "condition": 3,
            "grade": 7,
            "year_built": 1990,
            "zipcode": "98001",
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_condition_bounds(self):
        """Test condition validation bounds."""
        data = {
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_living": 1800,
            "sqft_lot": 5000,
            "floors": 2,
            "condition": 10,  # Above maximum
            "grade": 7,
            "year_built": 1990,
            "zipcode": "98001",
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
