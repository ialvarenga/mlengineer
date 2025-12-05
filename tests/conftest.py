"""
Pytest configuration and fixtures.

This file is loaded before any tests, ensuring environment variables
are mocked before app modules are imported.
"""

import os
import sys

# Set mock environment variables BEFORE any app imports
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing")
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "admin")
os.environ.setdefault("USE_AWS_SECRETS", "false")
os.environ.setdefault("USE_S3_MODEL", "false")
os.environ.setdefault("ENABLE_MODEL_AUTO_REFRESH", "false")
