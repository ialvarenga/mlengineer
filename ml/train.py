"""
Model Training Script

This script trains the housing price prediction model using the
provided dataset and saves the trained model to disk.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    HOUSING_DATA_PATH,
    ZIPCODE_DATA_PATH,
    ARTIFACTS_DIR,
    MODEL_METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from ml.model import HousingPriceModel


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load housing data and merge with demographic data.

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    logger.info("Loading housing data...")
    housing_df = pd.read_csv(HOUSING_DATA_PATH)

    logger.info("Loading demographic data...")
    demographics_df = pd.read_csv(ZIPCODE_DATA_PATH)

    # Convert zipcode to string for proper merging
    housing_df["zipcode"] = housing_df["zipcode"].astype(str)
    demographics_df["zipcode"] = demographics_df["zipcode"].astype(str)

    # Merge housing data with demographics
    logger.info("Merging housing data with demographics...")
    merged_df = housing_df.merge(
        demographics_df[
            ["zipcode", "median_income", "population_density", "median_age"]
        ],
        on="zipcode",
        how="left",
    )

    # Fill missing demographic data with median values
    for col in ["median_income", "population_density", "median_age"]:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())

    # Select features for the model
    feature_columns = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "condition",
        "grade",
        "year_built",
        "median_income",
        "population_density",
        "median_age",
    ]

    X = merged_df[feature_columns]
    y = merged_df["price"]

    logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[HousingPriceModel, dict]:
    """
    Train the model and evaluate its performance.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Create and train model
    model = HousingPriceModel(model_type="random_forest")
    model.fit(X_train, y_train)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "train": {
            "r2_score": r2_score(y_train, train_predictions),
            "rmse": np.sqrt(mean_squared_error(y_train, train_predictions)),
            "mae": mean_absolute_error(y_train, train_predictions),
        },
        "test": {
            "r2_score": r2_score(y_test, test_predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, test_predictions)),
            "mae": mean_absolute_error(y_test, test_predictions),
        },
        "n_samples": len(X),
        "n_features": len(X.columns),
        "feature_names": list(X.columns),
        "feature_importance": model.get_feature_importance(),
    }

    # Cross-validation
    cv_results = model.cross_validate(X, y, cv=5)
    metrics["cross_validation"] = cv_results

    # Log metrics
    logger.info("Model Training Complete!")
    logger.info(f"Train R² Score: {metrics['train']['r2_score']:.4f}")
    logger.info(f"Test R² Score: {metrics['test']['r2_score']:.4f}")
    logger.info(f"Test RMSE: ${metrics['test']['rmse']:,.2f}")
    logger.info(f"Test MAE: ${metrics['test']['mae']:,.2f}")
    logger.info(
        f"CV Mean R² Score: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score']:.4f})"
    )

    # Check for overfitting
    train_r2 = metrics["train"]["r2_score"]
    test_r2 = metrics["test"]["r2_score"]

    if train_r2 - test_r2 > 0.1:
        logger.warning(
            "⚠️ Potential OVERFITTING detected! Train-Test R² gap: {:.4f}".format(
                train_r2 - test_r2
            )
        )
        metrics["overfitting_warning"] = True
    elif test_r2 < 0.5:
        logger.warning(
            "⚠️ Potential UNDERFITTING detected! Test R² is low: {:.4f}".format(test_r2)
        )
        metrics["underfitting_warning"] = True
    else:
        logger.info("✅ Model appears well-fitted")
        metrics["overfitting_warning"] = False
        metrics["underfitting_warning"] = False

    return model, metrics


def save_metrics(metrics: dict):
    """Save model metrics to JSON file."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    metrics_serializable = convert_to_serializable(metrics)

    with open(MODEL_METRICS_PATH, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    logger.info(f"Metrics saved to {MODEL_METRICS_PATH}")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Housing Price Prediction Model Training")
    logger.info("=" * 60)

    # Load and prepare data
    X, y = load_and_prepare_data()

    # Train model
    model, metrics = train_model(X, y)

    # Save model
    model.save()

    # Save metrics
    save_metrics(metrics)

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    # Print feature importance
    logger.info("\nFeature Importance:")
    importance = metrics["feature_importance"]
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feature}: {score:.4f}")


if __name__ == "__main__":
    main()
