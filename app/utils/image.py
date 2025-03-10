import os
import uuid
import io
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, BinaryIO
from fastapi import UploadFile, HTTPException, status

from config import settings


async def validate_image(file: UploadFile) -> bool:
    """
    Validate that the uploaded file is an acceptable image
    
    Args:
        file: The uploaded file object
        
    Returns:
        bool: True if the image is valid
        
    Raises:
        HTTPException: If the image is invalid
    """
    # Check file size
    if file.size and file.size > settings.API.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the maximum allowed size of {settings.API.MAX_IMAGE_SIZE / (1024 * 1024)}MB"
        )
    
    # Check file type
    if file.content_type not in settings.API.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: {settings.API.ALLOWED_IMAGE_TYPES}"
        )
    
    # Attempt to open the image to validate it's not corrupted
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img.verify()  # Verify image is not corrupted
        
        # Reset file position for future reads
        await file.seek(0)
        return True
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )


async def save_upload_file(file: UploadFile) -> str:
    """
    Save an uploaded file to disk
    
    Args:
        file: The uploaded file object
        
    Returns:
        str: Path to the saved file
    """
    # Create a unique filename
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(settings.API.UPLOAD_DIR, filename)
    
    # Write file to disk
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Reset file position for future reads
    await file.seek(0)
    
    return file_path


async def read_image(file: UploadFile) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Read an image from an uploaded file and convert to OpenCV format
    
    Args:
        file: The uploaded file object
        
    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: The image as a numpy array and its original dimensions
    """
    contents = await file.read()
    image = np.asarray(bytearray(contents), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Reset file position for future reads
    await file.seek(0)
    
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image"
        )
    
    original_size = (image.shape[1], image.shape[0])  # width, height
    
    return image, original_size


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for model inference
    
    Args:
        image: The input image as a numpy array
        
    Returns:
        np.ndarray: The preprocessed image
    """
    # In a more advanced implementation, you might:
    # - Resize the image to the model's input size
    # - Normalize pixel values
    # - Convert BGR to RGB
    # - Apply any necessary transformations
    
    # For now, we'll keep it simple and let the YOLO model handle preprocessing
    return image


def draw_detections(image: np.ndarray, detections: list, thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image
    
    Args:
        image: The input image
        detections: List of detection results
        thickness: Line thickness for bounding boxes
        
    Returns:
        np.ndarray: Image with detections drawn
    """
    # Make a copy to avoid modifying the original
    result_image = image.copy()
    
    for detection in detections:
        # Extract bounding box
        x1, y1, x2, y2 = detection.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get class name and confidence
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        
        # Generate a color based on class_id (for consistency)
        color = (
            hash(detection.class_name) % 255,
            (hash(detection.class_name) * 57) % 255,
            (hash(detection.class_name) * 131) % 255,
        )
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            result_image,
            (x1, y1 - text_size[1] - 10),
            (x1 + text_size[0], y1),
            color,
            -1,
        )
        
        # Draw label text
        cv2.putText(
            result_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    
    return result_image