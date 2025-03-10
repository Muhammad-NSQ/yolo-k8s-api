from fastapi import Depends, HTTPException, status, UploadFile
from typing import Optional

from models import YOLOModel, get_model
from utils.image import validate_image

async def get_validated_model() -> YOLOModel:
    """
    Get the model instance and ensure it's loaded
    
    Returns:
        YOLOModel: The initialized YOLO model
        
    Raises:
        HTTPException: If the model is not loaded
    """
    model = get_model()
    if model.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded, service unavailable"
        )
    return model


async def validate_image_file(file: UploadFile) -> UploadFile:
    """
    Validate that an uploaded file is an acceptable image
    
    Args:
        file: The uploaded file to validate
        
    Returns:
        UploadFile: The validated file
        
    Raises:
        HTTPException: If the file is invalid
    """
    await validate_image(file)
    return file