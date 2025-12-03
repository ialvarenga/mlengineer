"""
Authentication module for the Housing Price Prediction API.

This module provides JWT-based authentication for securing endpoints.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import os

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 configuration for JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)


# Pydantic models for auth
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str


def get_jwt_settings():
    """Get JWT settings from environment."""
    return {
        "secret_key": os.getenv(
            "JWT_SECRET_KEY", "your-secret-key-change-in-production"
        ),
        "algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
        "expire_minutes": int(os.getenv("JWT_EXPIRATION_MINUTES", "30")),
    }


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user with username and password.

    For simplicity, this uses environment variables.
    In production, use a database.
    """
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin")

    if username == admin_username and password == admin_password:
        return User(username=username)
    return None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    settings = get_jwt_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings["expire_minutes"]
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings["secret_key"], algorithm=settings["algorithm"]
    )
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[User]:
    """Get the current user from a JWT token."""
    if not token:
        return None

    settings = get_jwt_settings()

    try:
        payload = jwt.decode(
            token, settings["secret_key"], algorithms=[settings["algorithm"]]
        )
        username = payload.get("sub")
        if username is None:
            return None
        return User(username=username)
    except JWTError:
        return None


async def require_auth(token: str = Depends(oauth2_scheme)) -> User:
    """
    Require a valid JWT token for authentication.

    Use this as a dependency on protected routes.

    Returns:
        The authenticated User.

    Raises:
        HTTPException: If the token is missing or invalid.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please login to get an access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_current_user(token)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please login again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
