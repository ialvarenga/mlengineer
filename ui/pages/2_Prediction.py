"""
Prediction Page for Housing Price Prediction API

Allows users to input house features and get price predictions.
"""

import streamlit as st
import requests
from typing import Optional

st.set_page_config(page_title="Prediction", page_icon="💰", layout="wide")


def init_session_state():
    """Initialize session state variables."""
    if "api_token" not in st.session_state:
        st.session_state.api_token = None
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"
    if "username" not in st.session_state:
        st.session_state.username = None


def get_auth_headers() -> Optional[dict]:
    """Get authorization headers if token is available."""
    if st.session_state.api_token:
        return {"Authorization": f"Bearer {st.session_state.api_token}"}
    return None


def predict_price(house_features: dict, use_minimal: bool = True) -> dict:
    """
    Make a prediction request to the API.

    Args:
        house_features: Dictionary of house features
        use_minimal: Whether to use the minimal endpoint

    Returns:
        dict with prediction results or error information
    """
    headers = get_auth_headers()
    if not headers:
        return {"success": False, "error": "Not authenticated"}

    endpoint = "/predict/minimal" if use_minimal else "/predict"

    try:
        response = requests.post(
            f"{st.session_state.api_base_url}{endpoint}",
            json=house_features,
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 401:
            st.session_state.api_token = None
            return {"success": False, "error": "Session expired. Please login again."}
        else:
            error_detail = response.json().get("detail", "Prediction failed")
            return {"success": False, "error": error_detail}

    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to the API server"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_model_info() -> dict:
    """Get information about the currently loaded model."""
    headers = get_auth_headers()
    if not headers:
        return {"success": False, "error": "Not authenticated"}

    try:
        response = requests.get(
            f"{st.session_state.api_base_url}/model/info",
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 401:
            return {"success": False, "error": "Session expired"}
        else:
            return {"success": False, "error": "Failed to get model info"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    init_session_state()

    st.title("House Price Prediction")

    if not st.session_state.api_token:
        st.warning("Please login first to make predictions.")
        st.page_link("pages/1_Login.py", label="Go to Login", icon="🔐")
        return

    st.sidebar.success(f"Logged in as: {st.session_state.username}")

    with st.sidebar:
        st.subheader("Model Information")
        if st.button("Refresh Model Info"):
            result = get_model_info()
            if result["success"]:
                st.session_state.model_info = result["data"]

        if "model_info" in st.session_state:
            st.json(st.session_state.model_info)

    st.subheader("Enter House Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Basic Information**")
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=20,
            value=3,
            help="Number of bedrooms",
        )
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.25,
            help="Number of bathrooms",
        )
        floors = st.number_input(
            "Floors",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.5,
            help="Number of floors",
        )
        zipcode = st.text_input(
            "Zipcode",
            value="98001",
            max_chars=10,
            help="5-digit zipcode for demographic data lookup",
        )

    with col2:
        st.markdown("**Size Information**")
        sqft_living = st.number_input(
            "Living Area (sqft)",
            min_value=100,
            max_value=50000,
            value=1800,
            step=100,
            help="Living area in square feet",
        )
        sqft_lot = st.number_input(
            "Lot Size (sqft)",
            min_value=100,
            max_value=2000000,
            value=5000,
            step=500,
            help="Lot size in square feet",
        )
        sqft_above = st.number_input(
            "Above Ground (sqft)",
            min_value=0,
            max_value=50000,
            value=1800,
            step=100,
            help="Square feet above ground",
        )
        sqft_basement = st.number_input(
            "Basement (sqft)",
            min_value=0,
            max_value=10000,
            value=0,
            step=100,
            help="Square feet of basement",
        )

    st.divider()

    if st.button("Get Price Prediction", type="primary", use_container_width=True):
        if len(zipcode) < 5:
            st.error("Please enter a valid 5-digit zipcode")
            return

        house_features = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "zipcode": zipcode,
        }

        with st.spinner("Getting prediction..."):
            result = predict_price(house_features, use_minimal=True)

        if result["success"]:
            prediction = result["data"]

            st.success("Prediction completed!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted Price",
                    f"${prediction['predicted_price']:,.2f}",
                )

            with col2:
                confidence = prediction.get("confidence_interval", {})
                lower = confidence.get("lower", 0)
                upper = confidence.get("upper", 0)
                st.metric(
                    "Price Range",
                    f"${lower:,.0f} - ${upper:,.0f}",
                )

            with col3:
                st.metric("Model Version", prediction.get("model_version", "N/A"))

            with st.expander("Demographic Data", expanded=True):
                demo_data = prediction.get("demographic_data", {})
                if demo_data:
                    dcol1, dcol2, dcol3 = st.columns(3)
                    with dcol1:
                        income = demo_data.get("median_household_income")
                        if income:
                            st.metric("Median Income", f"${income:,.0f}")
                    with dcol2:
                        pop = demo_data.get("population")
                        if pop:
                            st.metric("Population", f"{pop:,.0f}")
                    with dcol3:
                        housing_val = demo_data.get("housing_value")
                        if housing_val:
                            st.metric("Avg Housing Value", f"${housing_val:,.0f}")
                else:
                    st.info("No demographic data available for this zipcode")

            with st.expander("Full Response"):
                st.json(prediction)

        else:
            st.error(f"Prediction failed: {result['error']}")
            if "expired" in result["error"].lower():
                st.page_link("pages/1_Login.py", label="Go to Login", icon="🔐")


if __name__ == "__main__":
    main()
