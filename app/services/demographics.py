"""
Demographics Service

This service handles loading and looking up zip code demographic data.
The demographic data is automatically joined when making predictions.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ZIPCODE_DATA_PATH


class DemographicsService:
    """
    Service for managing and looking up zip code demographic data.
    """

    _instance = None
    _demographics_df = None

    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_demographics()
        return cls._instance

    def _load_demographics(self):
        """Load demographic data from CSV file."""
        try:
            self._demographics_df = pd.read_csv(ZIPCODE_DATA_PATH)
            self._demographics_df["zipcode"] = self._demographics_df["zipcode"].astype(
                str
            )
            logger.info(
                f"Loaded demographics data for {len(self._demographics_df)} zip codes"
            )
        except FileNotFoundError:
            logger.warning(f"Demographics file not found: {ZIPCODE_DATA_PATH}")
            self._demographics_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading demographics: {e}")
            self._demographics_df = pd.DataFrame()

    def get_demographics(self, zipcode: str) -> Dict:
        """
        Get demographic data for a specific zip code.

        Args:
            zipcode: The zip code to look up

        Returns:
            Dictionary with demographic data or default values
        """
        zipcode = str(zipcode).strip()

        if self._demographics_df is None or self._demographics_df.empty:
            logger.warning("Demographics data not loaded, returning defaults")
            return self._get_default_demographics(zipcode)

        # Look up the zipcode
        match = self._demographics_df[self._demographics_df["zipcode"] == zipcode]

        if match.empty:
            logger.warning(f"Zipcode {zipcode} not found, returning defaults")
            return self._get_default_demographics(zipcode)

        row = match.iloc[0]

        return {
            "zipcode": zipcode,
            "city": row.get("city", "Unknown"),
            "state": row.get("state", "Unknown"),
            "median_income": float(
                row.get("median_income", self._get_median("median_income"))
            ),
            "population_density": float(
                row.get("population_density", self._get_median("population_density"))
            ),
            "median_age": float(row.get("median_age", self._get_median("median_age"))),
            "unemployment_rate": float(row.get("unemployment_rate", 0)),
            "college_educated_pct": float(row.get("college_educated_pct", 0)),
            "median_home_value": float(row.get("median_home_value", 0)),
            "crime_rate": float(row.get("crime_rate", 0)),
        }

    def _get_default_demographics(self, zipcode: str) -> Dict:
        """Return default demographic values when zipcode is not found."""
        return {
            "zipcode": zipcode,
            "city": "Unknown",
            "state": "Unknown",
            "median_income": self._get_median("median_income"),
            "population_density": self._get_median("population_density"),
            "median_age": self._get_median("median_age"),
            "unemployment_rate": 0,
            "college_educated_pct": 0,
            "median_home_value": 0,
            "crime_rate": 0,
        }

    def _get_median(self, column: str) -> float:
        """Get median value for a column (used for missing data)."""
        if self._demographics_df is None or self._demographics_df.empty:
            # Default fallback values
            defaults = {
                "median_income": 65000,
                "population_density": 3000,
                "median_age": 35,
            }
            return defaults.get(column, 0)

        if column in self._demographics_df.columns:
            return float(self._demographics_df[column].median())
        return 0

    def get_model_features(self, zipcode: str) -> Dict[str, float]:
        """
        Get demographic features in the format expected by the model.

        Args:
            zipcode: The zip code to look up

        Returns:
            Dictionary with features for the model
        """
        demographics = self.get_demographics(zipcode)

        return {
            "median_income": demographics["median_income"],
            "population_density": demographics["population_density"],
            "median_age": demographics["median_age"],
        }

    def reload(self):
        """Reload demographic data from file."""
        self._load_demographics()
        logger.info("Demographics data reloaded")


# Global instance
demographics_service = DemographicsService()
