import os
import time
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO

from config import settings
from schemas import DetectionResult

# Initialize logger
logger = logging.getLogger(__name__)

class YOLOModel:
    """YOLO model wrapper for object detection"""
    
    def __init__(self):
        """Initialize the YOLO model"""
        self.model = None
        self.model_name = settings.MODEL.MODEL_NAME
        self.device = settings.MODEL.DEVICE
        self.conf_threshold = settings.MODEL.CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.MODEL.IOU_THRESHOLD
        self.max_detections = settings.MODEL.MAX_DETECTIONS
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            start_time = time.time()
            
            # Set environment variable to bypass the security check
            os.environ["ULTRALYTICS_SKIP_TORCH_WEIGHTS_WARNING"] = "1"
            
            # Configure torch serialization for security
            try:
                # For newer torch versions
                import torch.serialization
                torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            except (ImportError, AttributeError):
                # Fall back to environment variable solution
                pass
            
            # Load the model
            self.model = YOLO(self.model_name)
            
            # Move model to specified device (CPU/GPU)
            self.model.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
    
    def predict(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run inference on an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List[DetectionResult]: Detection results
        """
        if self.model is None:
            logger.error("Model not loaded")
            raise RuntimeError("Model not loaded")
        
        # Measure inference time
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model.predict(
                image, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            # Process results
            detections = self._process_results(results[0])  # Get first result (batch size=1)
            
            # Log performance
            inference_time = time.time() - start_time
            logger.debug(f"Inference completed in {inference_time:.4f} seconds with {len(detections)} detections")
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def _process_results(self, result) -> List[DetectionResult]:
        """
        Process YOLO results into standardized schema
        
        Args:
            result: YOLO result object
            
        Returns:
            List[DetectionResult]: Standardized detection results
        """
        detections = []
        
        # Extract boxes, confidence scores and class IDs
        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
        confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else []
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes, 'cls') else []
        
        # Get class names mapping
        names = result.names
        
        # Create detection results
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = float(confs[i])
            class_id = int(class_ids[i])
            class_name = names[class_id]
            
            detections.append(
                DetectionResult(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=[float(x1), float(y1), float(x2), float(y2)]
                )
            )
        
        return detections

# Global model instance
_model_instance = None

def get_model() -> YOLOModel:
    """
    Get or create the model instance
    
    Returns:
        YOLOModel: The YOLO model instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = YOLOModel()
    
    return _model_instance