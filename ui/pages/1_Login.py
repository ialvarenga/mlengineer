"""
Login Page for Housing Price Prediction API

Handles user authentication and stores the JWT token for subsequent requests.
"""

import streamlit as st
import requests

st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")


def init_session_state():
    """Initialize session state variables."""
    if "api_token" not in st.session_state:
        st.session_state.api_token = None
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"
    if "username" not in st.session_state:
        st.session_state.username = None


def login(base_url: str, username: str, password: str) -> dict:
    """
    Authenticate with the API and get an access token.

    Args:
        base_url: API base URL
        username: Username for authentication
        password: Password for authentication

    Returns:
        dict with access_token and token_type, or error information
    """
    try:
        response = requests.post(
            f"{base_url}/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            error_detail = response.json().get("detail", "Authentication failed")
            return {"success": False, "error": error_detail}

    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to the API server"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_api_health(base_url: str) -> dict:
    """Check if the API is healthy and accessible."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        return {
            "success": False,
            "error": f"API returned status {response.status_code}",
        }
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to the API server"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    init_session_state()

    st.title("Login")

    if st.session_state.api_token:
        st.success(f"You are already logged in as **{st.session_state.username}**")
        st.info(f"API URL: {st.session_state.api_base_url}")

        if st.button("Logout"):
            st.session_state.api_token = None
            st.session_state.username = None
            st.rerun()

        st.divider()
        st.subheader("Change API Settings")

    st.subheader("API Configuration")

    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="The base URL of the Housing Price Prediction API",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Check API Health"):
            with st.spinner("Checking API health..."):
                result = check_api_health(api_url)
                if result["success"]:
                    health_data = result["data"]
                    st.success("API is healthy!")
                    st.json(health_data)
                else:
                    st.error(f"API check failed: {result['error']}")

    if api_url != st.session_state.api_base_url:
        st.session_state.api_base_url = api_url

    st.divider()

    if not st.session_state.api_token:
        st.subheader("Credentials")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password"
            )

            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    with st.spinner("Authenticating..."):
                        result = login(api_url, username, password)

                        if result["success"]:
                            token_data = result["data"]
                            st.session_state.api_token = token_data["access_token"]
                            st.session_state.username = username
                            st.success("Login successful!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"Login failed: {result['error']}")

        st.info(
            """
            **Default credentials:**
            - Username: admin
            - Password: admin
            
            *These can be changed via environment variables on the API server.*
            """
        )


if __name__ == "__main__":
    main()
