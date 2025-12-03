"""
Model Watcher Service

Background service that periodically checks S3 for new model versions
and hot-reloads the model when a newer version is available.
"""
import asyncio
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    USE_S3_MODEL,
    S3_MODEL_BUCKET,
    S3_MODEL_PREFIX,
    AWS_REGION,
    MODEL_REFRESH_INTERVAL_SECONDS,
    ENABLE_MODEL_AUTO_REFRESH,
)


class ModelWatcher:
    """
    Watches S3 for new model versions and triggers hot-reload when found.
    """
    
    _instance = None
    _is_running: bool = False
    _task: Optional[asyncio.Task] = None
    _last_check: Optional[datetime] = None
    _check_count: int = 0
    _reload_count: int = 0
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_s3_client(self):
        """Get boto3 S3 client."""
        import boto3
        return boto3.client('s3', region_name=AWS_REGION)
    
    def _get_latest_version_from_s3(self) -> Optional[str]:
        """Get the latest model version from S3 latest.txt pointer."""
        try:
            s3_client = self._get_s3_client()
            latest_key = f"{S3_MODEL_PREFIX}/latest.txt"
            response = s3_client.get_object(Bucket=S3_MODEL_BUCKET, Key=latest_key)
            version = response['Body'].read().decode('utf-8').strip()
            return version
        except Exception as e:
            logger.error(f"Failed to get latest version from S3: {e}")
            return None
    
    async def check_and_reload(self) -> dict:
        """
        Check S3 for new model version and reload if needed.
        
        Returns:
            dict with check result info
        """
        from app.services.prediction import prediction_service
        
        self._last_check = datetime.now(timezone.utc)
        self._check_count += 1
        
        result = {
            "checked_at": self._last_check.isoformat(),
            "check_number": self._check_count,
            "action": "none",
            "current_version": prediction_service._loaded_version,
            "latest_version": None,
            "reloaded": False,
        }
        
        if not USE_S3_MODEL:
            result["action"] = "skipped"
            result["reason"] = "S3 model loading is disabled (USE_S3_MODEL=false)"
            logger.debug("Model check skipped: S3 model loading is disabled")
            return result
        
        # Get latest version from S3
        latest_version = self._get_latest_version_from_s3()
        result["latest_version"] = latest_version
        
        if not latest_version:
            result["action"] = "error"
            result["reason"] = "Could not retrieve latest version from S3"
            return result
        
        current_version = prediction_service._loaded_version
        
        # Compare versions (they are timestamps, so string comparison works)
        if current_version == latest_version:
            result["action"] = "no_update"
            result["reason"] = f"Current version {current_version} is already the latest"
            logger.debug(f"Model check: version {current_version} is current")
            return result
        
        # New version available - reload
        logger.info(f"New model version detected: {latest_version} (current: {current_version})")
        
        try:
            # Reload the model
            prediction_service.reload_model()
            self._reload_count += 1
            
            result["action"] = "reloaded"
            result["reloaded"] = True
            result["new_version"] = prediction_service._loaded_version
            result["reason"] = f"Successfully reloaded from {current_version} to {prediction_service._loaded_version}"
            
            logger.info(f"Model hot-reloaded to version: {prediction_service._loaded_version}")
            
        except Exception as e:
            result["action"] = "error"
            result["reason"] = f"Failed to reload model: {str(e)}"
            logger.error(f"Failed to reload model: {e}")
        
        return result
    
    async def _watch_loop(self):
        """Background loop that periodically checks for model updates."""
        logger.info(f"Model watcher started (interval: {MODEL_REFRESH_INTERVAL_SECONDS}s)")
        
        while self._is_running:
            try:
                await self.check_and_reload()
            except Exception as e:
                logger.error(f"Error in model watcher loop: {e}")
            
            # Wait for next check interval
            await asyncio.sleep(MODEL_REFRESH_INTERVAL_SECONDS)
    
    async def start(self):
        """Start the background model watcher task."""
        if not ENABLE_MODEL_AUTO_REFRESH:
            logger.info("Model auto-refresh is disabled (ENABLE_MODEL_AUTO_REFRESH=false)")
            return
        
        if not USE_S3_MODEL:
            logger.info("Model watcher not started: S3 model loading is disabled")
            return
        
        if self._is_running:
            logger.warning("Model watcher is already running")
            return
        
        self._is_running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("Model watcher background task started")
    
    async def stop(self):
        """Stop the background model watcher task."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info("Model watcher stopped")
    
    def get_status(self) -> dict:
        """Get current status of the model watcher."""
        from app.services.prediction import prediction_service
        
        return {
            "enabled": ENABLE_MODEL_AUTO_REFRESH and USE_S3_MODEL,
            "is_running": self._is_running,
            "refresh_interval_seconds": MODEL_REFRESH_INTERVAL_SECONDS,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "total_checks": self._check_count,
            "total_reloads": self._reload_count,
            "current_model_version": prediction_service._loaded_version,
            "s3_bucket": S3_MODEL_BUCKET,
            "s3_prefix": S3_MODEL_PREFIX,
        }


# Global instance
model_watcher = ModelWatcher()
