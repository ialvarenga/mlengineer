"""
Streamlit UI for Housing Price Prediction API

Main application entry point that provides navigation between pages.
"""

import streamlit as st

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    if "api_token" not in st.session_state:
        st.session_state.api_token = None
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"
    if "username" not in st.session_state:
        st.session_state.username = None


def main():
    init_session_state()

    st.sidebar.title("Navigation")

    if st.session_state.api_token:
        st.sidebar.success(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.api_token = None
            st.session_state.username = None
            st.rerun()

    st.title("Housing Price Prediction")
    st.markdown(
        """
        Welcome to the Housing Price Prediction application.
        
        **Features:**
        - Predict house prices based on features and demographics
        - Batch predictions for multiple houses
        - View model information
        
        **Getting Started:**
        1. Go to the **Login** page to authenticate
        2. Use the **Prediction** page to get price estimates
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Authentication Status",
            "Authenticated" if st.session_state.api_token else "Not Authenticated",
        )

    with col2:
        st.metric("API URL", st.session_state.api_base_url)

    with col3:
        if st.session_state.username:
            st.metric("User", st.session_state.username)


if __name__ == "__main__":
    main()
