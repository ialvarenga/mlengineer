"""
Batch Prediction Page for Housing Price Prediction API

Allows users to upload a CSV file and get predictions for multiple houses.
"""

import streamlit as st
import requests
import pandas as pd
from io import StringIO
from typing import Optional

st.set_page_config(page_title="Batch Prediction", page_icon="📊", layout="wide")


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


def predict_batch(houses: list) -> dict:
    """
    Make a batch prediction request to the API.

    Args:
        houses: List of house feature dictionaries

    Returns:
        dict with prediction results or error information
    """
    headers = get_auth_headers()
    if not headers:
        return {"success": False, "error": "Not authenticated"}

    try:
        response = requests.post(
            f"{st.session_state.api_base_url}/predict/batch",
            json={"houses": houses},
            headers=headers,
            timeout=120,
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 401:
            st.session_state.api_token = None
            return {"success": False, "error": "Session expired. Please login again."}
        else:
            error_detail = response.json().get("detail", "Batch prediction failed")
            return {"success": False, "error": error_detail}

    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to the API server"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    init_session_state()

    st.title("Batch House Price Prediction")

    if not st.session_state.api_token:
        st.warning("Please login first to make predictions.")
        st.page_link("pages/1_Login.py", label="Go to Login", icon="🔐")
        return

    st.sidebar.success(f"Logged in as: {st.session_state.username}")

    st.markdown(
        """
        Upload a CSV file with house features to get price predictions for multiple houses at once.
        
        **Required columns:**
        - bedrooms, bathrooms, sqft_living, sqft_lot, floors
        - sqft_above, sqft_basement, zipcode
        
        **Optional columns:**
        - waterfront, view, condition, grade, yr_built, yr_renovated
        - lat, long, sqft_living15, sqft_lot15
        """
    )

    sample_data = pd.DataFrame(
        {
            "bedrooms": [3, 4, 2],
            "bathrooms": [2.0, 2.5, 1.0],
            "sqft_living": [1800, 2500, 1200],
            "sqft_lot": [5000, 7500, 4000],
            "floors": [2.0, 2.0, 1.0],
            "sqft_above": [1800, 2000, 1200],
            "sqft_basement": [0, 500, 0],
            "zipcode": ["98001", "98002", "98003"],
        }
    )

    with st.expander("View Sample CSV Format"):
        st.dataframe(sample_data)
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV",
            data=csv,
            file_name="sample_houses.csv",
            mime="text/csv",
        )

    st.divider()

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")
            st.dataframe(df, use_container_width=True)

            required_columns = [
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "sqft_lot",
                "floors",
                "sqft_above",
                "sqft_basement",
                "zipcode",
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return

            df["zipcode"] = df["zipcode"].astype(str)

            st.info(f"Found {len(df)} houses to predict")

            if st.button(
                "Run Batch Prediction", type="primary", use_container_width=True
            ):
                houses = df[required_columns].to_dict("records")

                with st.spinner(f"Getting predictions for {len(houses)} houses..."):
                    result = predict_batch(houses)

                if result["success"]:
                    predictions = result["data"]["predictions"]
                    total_count = result["data"]["total_count"]

                    st.success(
                        f"Successfully predicted prices for {total_count} houses!"
                    )

                    results_df = df.copy()
                    results_df["predicted_price"] = [
                        p["predicted_price"] for p in predictions
                    ]
                    results_df["price_lower"] = [
                        p["confidence_interval"]["lower"] for p in predictions
                    ]
                    results_df["price_upper"] = [
                        p["confidence_interval"]["upper"] for p in predictions
                    ]

                    st.subheader("Prediction Results")
                    st.dataframe(
                        results_df.style.format(
                            {
                                "predicted_price": "${:,.2f}",
                                "price_lower": "${:,.2f}",
                                "price_upper": "${:,.2f}",
                            }
                        ),
                        use_container_width=True,
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Average Predicted Price",
                            f"${results_df['predicted_price'].mean():,.2f}",
                        )

                    with col2:
                        st.metric(
                            "Price Range",
                            f"${results_df['predicted_price'].min():,.0f} - ${results_df['predicted_price'].max():,.0f}",
                        )

                    csv_output = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv_output,
                        file_name="prediction_results.csv",
                        mime="text/csv",
                    )

                    with st.expander("Full API Response"):
                        st.json(result["data"])

                else:
                    st.error(f"Batch prediction failed: {result['error']}")
                    if "expired" in result["error"].lower():
                        st.page_link("pages/1_Login.py", label="Go to Login", icon="🔐")

        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")


if __name__ == "__main__":
    main()
