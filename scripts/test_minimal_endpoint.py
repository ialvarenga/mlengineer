#!/usr/bin/env python
"""
Test Script for Minimal Prediction Endpoint

Demonstrates the /predict/minimal endpoint using examples from future_unseen_examples.csv.
This endpoint only requires the features that the ML model actually uses.
"""
import os
import csv
import requests
from typing import Optional


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
USERNAME = os.getenv("API_USERNAME", "admin")
PASSWORD = os.getenv("API_PASSWORD", "admin")
DATA_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "future_unseen_examples.csv"
)

_access_token: Optional[str] = None


def login() -> Optional[str]:
    """Login and get an access token."""
    global _access_token
    try:
        response = requests.post(
            f"{API_BASE_URL}/login",
            data={"username": USERNAME, "password": PASSWORD},
            timeout=10,
        )
        if response.status_code == 200:
            _access_token = response.json()["access_token"]
            print("Login successful")
            return _access_token
        else:
            print(f"Login failed: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to API at {API_BASE_URL}")
        print("Make sure the server is running: python -m uvicorn app.main:app")
        return None


def predict_minimal(house_data: dict) -> Optional[dict]:
    """Call the minimal prediction endpoint."""
    headers = {"Authorization": f"Bearer {_access_token}"}
    response = requests.post(
        f"{API_BASE_URL}/predict/minimal",
        json=house_data,
        headers=headers,
        timeout=10,
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Prediction failed: {response.status_code} - {response.text}")
        return None


def load_examples(num_examples: int = 5) -> list:
    """Load examples from future_unseen_examples.csv."""
    examples = []
    with open(DATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_examples:
                break
            examples.append(
                {
                    "bedrooms": int(row["bedrooms"]),
                    "bathrooms": float(row["bathrooms"]),
                    "sqft_living": int(row["sqft_living"]),
                    "sqft_lot": int(row["sqft_lot"]),
                    "floors": float(row["floors"]),
                    "sqft_above": int(row["sqft_above"]),
                    "sqft_basement": int(row["sqft_basement"]),
                    "zipcode": row["zipcode"],
                }
            )
    return examples


def main():
    print("=" * 60)
    print("Testing /predict/minimal endpoint")
    print("=" * 60)

    if not login():
        return

    examples = load_examples(5)
    print(f"\nLoaded {len(examples)} examples from {DATA_FILE}\n")

    for i, house in enumerate(examples, 1):
        print(f"--- Example {i} ---")
        print(
            f"Input: {house['bedrooms']} bed, {house['bathrooms']} bath, "
            f"{house['sqft_living']} sqft, zipcode {house['zipcode']}"
        )

        result = predict_minimal(house)
        if result:
            price = result["predicted_price"]
            lower = result["confidence_interval"]["lower"]
            upper = result["confidence_interval"]["upper"]
            print(f"Predicted Price: ${price:,.2f}")
            print(f"Confidence Interval: ${lower:,.2f} - ${upper:,.2f}")
        print()

    print("Test completed")


if __name__ == "__main__":
    main()
