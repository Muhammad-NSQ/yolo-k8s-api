import os
import cv2
import time
import logging
import tempfile
import numpy as np
from typing import Tuple, Generator, Optional, Dict, Any, List
from pathlib import Path

from fastapi import UploadFile, HTTPException, status
from config import settings
from schemas import DetectionResult
from utils.image import draw_detections

# Initialize logger
logger = logging.getLogger(__name__)

async def validate_video(file: UploadFile) -> bool:
    """
    Validate that the uploaded file is an acceptable video
    
    Args:
        file: The uploaded file object
        
    Returns:
        bool: True if the video is valid
        
    Raises:
        HTTPException: If the video is invalid
    """
    # Check file size
    if file.size and file.size > settings.API.MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the maximum allowed size of {settings.API.MAX_VIDEO_SIZE / (1024 * 1024)}MB"
        )
    
    # Check file type
    if file.content_type not in settings.API.ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: {settings.API.ALLOWED_VIDEO_TYPES}"
        )
    
    # Save the file temporarily to verify it's a valid video
    temp_file = None
    try:
        contents = await file.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(contents)
        temp_file.close()
        
        # Try to open with OpenCV to verify it's a valid video
        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid video file: Cannot open video"
            )
        
        # Read first frame to verify
        ret, _ = cap.read()
        if not ret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid video file: Cannot read frames"
            )
            
        # Get basic video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Valid video: {width}x{height} at {fps} FPS, {frame_count} frames")
        
        # Close the video
        cap.release()
        
        # Reset file position for future reads
        await file.seek(0)
        return True
        
    except Exception as e:
        logger.error(f"Error validating video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid video file: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


async def save_video_file(file: UploadFile) -> str:
    """
    Save an uploaded video file to disk
    
    Args:
        file: The uploaded file object
        
    Returns:
        str: Path to the saved file
    """
    # Create a unique filename
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    filename = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(4).hex()}{file_ext}"
    file_path = os.path.join(settings.API.UPLOAD_DIR, filename)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write file to disk
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Reset file position for future reads
    await file.seek(0)
    
    return file_path


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get information about a video file
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dict[str, Any]: Video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Release the video
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
        "file_size": os.path.getsize(video_path)
    }


def frame_generator(video_path: str, max_frames: Optional[int] = None, 
                   skip_frames: int = 0, start_time: float = 0, 
                   end_time: Optional[float] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generate frames from a video file
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to process (None for all)
        skip_frames: Number of frames to skip between processed frames
        start_time: Start time in seconds
        end_time: End time in seconds (None for end of video)
        
    Yields:
        Tuple[int, np.ndarray]: Frame index and frame image
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Calculate start and end frames
    start_frame = int(start_time * fps) if start_time > 0 else 0
    end_frame = int(end_time * fps) if end_time is not None else frame_count
    
    # Ensure valid range
    start_frame = max(0, min(start_frame, frame_count - 1))
    end_frame = max(start_frame, min(end_frame, frame_count))
    
    # Set starting position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    try:
        # Process frames
        frame_idx = start_frame
        frames_processed = 0
        
        while frame_idx < end_frame:
            # Read frame
            ret, frame = cap.read()
            
            # Break if end of video or error
            if not ret:
                break
            
            # Process frame if not skipping
            if frames_processed % (skip_frames + 1) == 0:
                yield frame_idx, frame
            
            # Update counters
            frame_idx += 1
            frames_processed += 1
            
            # Break if reached max frames
            if max_frames is not None and frames_processed >= max_frames * (skip_frames + 1):
                break
                
    finally:
        # Release the video
        cap.release()


def create_result_video(video_path: str, detections_by_frame: Dict[int, List[DetectionResult]], 
                       output_path: Optional[str] = None, fps: Optional[float] = None) -> str:
    """
    Create a result video with detections drawn on frames
    
    Args:
        video_path: Path to the input video
        detections_by_frame: Dictionary mapping frame indices to detection results
        output_path: Path to save the output video (None for auto-generate)
        fps: FPS for output video (None to use source FPS)
        
    Returns:
        str: Path to the result video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use source FPS if not specified
    if fps is None:
        fps = src_fps
    
    # Create output path if not specified
    if output_path is None:
        # Get the results directory with fallback
        results_dir = getattr(settings.API, "RESULTS_DIR", os.path.join(settings.API.UPLOAD_DIR, "results"))
        # Make sure the directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        input_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(
            results_dir,
            f"{input_name}_detected_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        )
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        # Process each frame
        for frame_idx in range(frame_count):
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Draw detections if available for this frame
            if frame_idx in detections_by_frame:
                frame = draw_detections(frame, detections_by_frame[frame_idx])
                
            # Write frame to output video
            out.write(frame)
            
        return output_path
        
    finally:
        # Release resources
        cap.release()
        out.release()