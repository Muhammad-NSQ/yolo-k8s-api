from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class ModelSettings(BaseSettings):
    """Settings for YOLO model configuration"""
    MODEL_NAME: str = "yolov8l.pt"  # Default to YOLOv8 nano
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    USE_TRITON: bool = False  # Default to direct inference
    TRITON_URL: str = "triton:8001"  # Default gRPC endpoint
    DEVICE: str = "cuda:0"  # Use 'cuda:0' for GPU
    MAX_DETECTIONS: int = 100
    
    # TensorRT settings
    USE_TENSORRT: bool = False  # Enable TensorRT acceleration
    TENSORRT_FP16: bool = True  # Use FP16 precision (faster, slightly less accurate)
    TENSORRT_CACHE_DIR: str = "/tmp/tensorrt"  # Directory to store TensorRT engine files

    class Config:
        env_prefix = "MODEL_"


class APISettings(BaseSettings):
    """API configuration settings"""
    DEBUG: bool = False
    TITLE: str = "YOLO Object Detection API"
    DESCRIPTION: str = "API for real-time object detection using YOLO models"
    VERSION: str = "0.1.0"
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: List[str] = ["*"]
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png"]
    UPLOAD_DIR: str = "/tmp/uploads"
    RESULTS_TTL: int = 3600  # Time to live for cached results in seconds

    class Config:
        env_prefix = "API_"


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    LEVEL: str = "INFO"
    JSON_FORMAT: bool = True  # Structured logging in JSON format
    ENABLE_ACCESS_LOG: bool = True

    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """Main settings class that includes all subconfigurations"""
    API: APISettings = APISettings()
    MODEL: ModelSettings = ModelSettings()
    LOGGING: LoggingSettings = LoggingSettings()

    class Config:
        env_nested_delimiter = "__"


def get_settings() -> Settings:
    """Get application settings with environment variables applied"""
    return Settings()


# Singleton instance for application-wide access
settings = get_settings()


# Ensure upload directory exists
os.makedirs(settings.API.UPLOAD_DIR, exist_ok=True)

# Ensure TensorRT cache directory exists if TensorRT is enabled
if settings.MODEL.USE_TENSORRT:
    os.makedirs(settings.MODEL.TENSORRT_CACHE_DIR, exist_ok=True)