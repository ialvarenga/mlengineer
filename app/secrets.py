"""
AWS Secrets Manager integration for loading application secrets.

This module handles fetching secrets from AWS Secrets Manager at application startup.
"""

import json
import os
from typing import Optional
from loguru import logger

# Flag to track if secrets have been loaded
_secrets_loaded = False


def get_secrets_manager_client():
    """Get boto3 Secrets Manager client."""
    try:
        import boto3

        region = os.getenv("AWS_REGION", "us-east-1")
        return boto3.client("secretsmanager", region_name=region)
    except ImportError:
        logger.warning("boto3 not installed. AWS Secrets Manager integration disabled.")
        return None


def load_secrets_from_aws(secret_name: Optional[str] = None) -> bool:
    """
    Load secrets from AWS Secrets Manager and set them as environment variables.

    The secret should be a JSON object with keys:
    - JWT_SECRET_KEY: Secret key for signing JWT tokens
    - ADMIN_PASSWORD: Password for the admin user
    - ADMIN_USERNAME: (optional) Username for the admin user

    Args:
        secret_name: Name of the secret in AWS Secrets Manager.
                    Defaults to AWS_SECRET_NAME env var or "housing-api/secrets".

    Returns:
        True if secrets were loaded successfully, False otherwise.
    """
    global _secrets_loaded

    if _secrets_loaded:
        logger.debug("Secrets already loaded, skipping.")
        return True

    # Get secret name from parameter or environment
    secret_name = secret_name or os.getenv("AWS_SECRET_NAME", "housing-api/secrets")

    # Check if we should use AWS Secrets Manager (default: true)
    use_aws_secrets = os.getenv("USE_AWS_SECRETS", "true").lower() == "true"

    if not use_aws_secrets:
        logger.info(
            "AWS Secrets Manager disabled (USE_AWS_SECRETS != 'true'). Using environment variables."
        )
        return False

    client = get_secrets_manager_client()
    if not client:
        logger.warning(
            "Could not create Secrets Manager client. Using environment variables."
        )
        return False

    try:
        logger.info(f"Loading secrets from AWS Secrets Manager: {secret_name}")

        response = client.get_secret_value(SecretId=secret_name)

        # Parse the secret string as JSON
        if "SecretString" in response:
            secrets = json.loads(response["SecretString"])
        else:
            # Binary secret - decode it
            import base64

            secrets = json.loads(base64.b64decode(response["SecretBinary"]))

        # Set environment variables from secrets
        secret_keys = ["JWT_SECRET_KEY", "ADMIN_PASSWORD", "ADMIN_USERNAME"]
        loaded_count = 0

        for key in secret_keys:
            if key in secrets:
                os.environ[key] = secrets[key]
                loaded_count += 1
                logger.debug(f"Loaded secret: {key}")

        _secrets_loaded = True
        logger.info(
            f"✅ Successfully loaded {loaded_count} secrets from AWS Secrets Manager"
        )
        return True

    except client.exceptions.ResourceNotFoundException:
        logger.error(f"Secret '{secret_name}' not found in AWS Secrets Manager")
        return False
    except Exception as e:
        error_msg = str(e)
        if "AccessDenied" in error_msg:
            logger.error(
                f"Access denied to secret '{secret_name}'. Check IAM permissions."
            )
        elif (
            "NoCredentialsError" in error_msg
            or "Unable to locate credentials" in error_msg
        ):
            logger.error(
                "AWS credentials not found. Configure credentials via environment variables or AWS CLI."
            )
        else:
            logger.error(f"Failed to load secrets from AWS: {e}")
        return False


def is_secrets_loaded() -> bool:
    """Check if secrets have been loaded from AWS."""
    return _secrets_loaded
