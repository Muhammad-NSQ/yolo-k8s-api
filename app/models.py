import os
import time
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO
from pathlib import Path

from config import settings
from schemas import DetectionResult

# Initialize logger
logger = logging.getLogger(__name__)

class YOLOModel:
    """YOLO model wrapper for object detection with ONNX optimization"""
    
    def __init__(self):
        """Initialize the YOLO model"""
        self.model = None
        self.model_name = settings.MODEL.MODEL_NAME
        self.device = settings.MODEL.DEVICE
        self.conf_threshold = settings.MODEL.CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.MODEL.IOU_THRESHOLD
        self.max_detections = settings.MODEL.MAX_DETECTIONS
        self.use_optimization = settings.MODEL.USE_TENSORRT  # Flag for TensorRT optimization
        self.use_fp16 = settings.MODEL.TENSORRT_FP16
        self.optimize_dir = settings.MODEL.TENSORRT_CACHE_DIR
        
        # Check CUDA availability first and adjust device if needed
        try:
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning(f"CUDA requested but not available. Falling back to CPU.")
                self.device = 'cpu'
                self.use_optimization = False
        except Exception as e:
            logger.warning(f"Error checking CUDA: {str(e)}. Falling back to CPU.")
            self.device = 'cpu'
            self.use_optimization = False
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model with ONNX optimization if available"""
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            start_time = time.time()
            
            # Set environment variable to bypass the security check
            os.environ["ULTRALYTICS_SKIP_TORCH_WEIGHTS_WARNING"] = "1"
            
            # Check for pre-exported ONNX model if optimization is enabled
            if self.use_optimization and self.device.startswith('cuda'):
                logger.info("TensorRT optimization enabled, checking for ONNX model")
                
                # Determine path to optimized model
                model_basename = Path(self.model_name).stem
                onnx_path = Path(self.optimize_dir) / f"{model_basename}_fp{'16' if self.use_fp16 else '32'}.onnx"
                
                if onnx_path.exists():
                    logger.info(f"Loading optimized ONNX model: {onnx_path}")
                    try:
                        # For ONNX models with TensorRT, specify the device during loading
                        self.model = YOLO(str(onnx_path))
                        logger.info("ONNX model loaded successfully with TensorRT optimization")
                    except Exception as e:
                        logger.error(f"Failed to load ONNX model: {str(e)}")
                        logger.warning("Falling back to standard PyTorch model")
                        self.model = YOLO(self.model_name)
                        self.model.to(self.device)
                else:
                    logger.info(f"Optimized ONNX model not found at {onnx_path}")
                    logger.info("Loading standard PyTorch model")
                    self.model = YOLO(self.model_name)
                    self.model.to(self.device)
            else:
                # Load standard PyTorch model
                logger.info("Loading standard PyTorch model")
                self.model = YOLO(self.model_name)
                logger.info(f"Moving model to device: {self.device}")
                self.model.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            
            # Attempt to fall back to CPU if GPU loading failed
            if self.device != 'cpu':
                logger.info("Attempting to fall back to CPU")
                try:
                    self.device = 'cpu'
                    self.model = YOLO(self.model_name)
                    self.model.to(self.device)
                    logger.info("Successfully loaded model on CPU")
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {str(e2)}")
                    raise RuntimeError(f"Failed to load YOLO model: {str(e)}, CPU fallback also failed: {str(e2)}")
            else:
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
            with torch.no_grad():
                # Check if model is ONNX
                model_path = getattr(self.model, "ckpt_path", "")
                is_onnx = str(model_path).endswith(".onnx")
                is_fp16 = self.use_fp16 and is_onnx
                
                # Prediction args
                pred_args = {
                    'conf': self.conf_threshold,
                    'iou': self.iou_threshold,
                    'max_det': self.max_detections,
                    'verbose': False
                }
                
                # For ONNX models, specify device and data type for TensorRT
                if is_onnx:
                    pred_args['device'] = self.device
                    
                    # Explicitly handle FP16 for ONNX with TensorRT
                    if is_fp16:
                        pred_args['half'] = True
                
                # Run prediction
                results = self.model.predict(image, **pred_args)
            
            # Process results
            detections = self._process_results(results[0])
            
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict[str, Any]: Model information
        """
        gpu_available = False
        gpu_name = None
        
        try:
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
        except:
            pass
            
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "optimization_enabled": self.use_optimization,
            "fp16_enabled": self.use_fp16 if self.use_optimization else None,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
        }
        
        # Add more detailed info if model is loaded
        if self.model is not None:
            info["task"] = getattr(self.model, "task", "unknown")
            
            # Check if model is ONNX
            model_path = getattr(self.model, "ckpt_path", "")
            info["model_format"] = "ONNX" if str(model_path).endswith(".onnx") else "PyTorch"
            
            # Add class names if available
            if hasattr(self.model, "names"):
                info["classes"] = self.model.names
        
        return info

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