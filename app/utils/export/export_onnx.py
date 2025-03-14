#!/usr/bin/env python3
"""
Script to export YOLOv8 models to ONNX format for optimized inference
"""
import os
import argparse
import torch
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path

def export_to_onnx(model_path, output_dir, device='cuda:0', fp16=True):
    """
    Export a YOLOv8 model to ONNX format.

    Args:
        model_path: Path to the YOLOv8 model (.pt file)
        output_dir: Directory to save the exported model
        device: Device to use for export (e.g., 'cuda:0' for first GPU, 'cpu' for CPU)
        fp16: Whether to use FP16 precision
    
    Returns:
        str: Path to the exported model
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if CUDA is available when GPU is requested
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU.")
        device = "cpu"

    # Load YOLO model
    print(f"🚀 Loading model: {model_path} on {device}")
    model = YOLO(model_path)

    # Define ONNX export arguments
    export_args = {
        'format': 'onnx',
        'dynamic': False,  # Static shapes for better performance
        'half': fp16,  # Use FP16 precision
        'device': device,  # Use specified device
        'simplify': True,  # Simplify model
        'opset': 12,  # ONNX opset version
        'verbose': True,  # Show verbose output
    }

    # Export model to ONNX
    output_path = model.export(**export_args)
    print(f"✅ Model exported to: {output_path}")

    # Move to output directory if needed
    model_name = Path(model_path).stem
    output_file = Path(output_dir) / f"{model_name}_fp{'16' if fp16 else '32'}.onnx"

    if str(output_path) != str(output_file):
        import shutil
        shutil.copy(output_path, output_file)
        print(f"📂 Model copied to: {output_file}")

    # Validate ONNX model with CUDA execution provider
    try:
        session = ort.InferenceSession(str(output_file), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        print(f"✅ ONNX model successfully loaded with CUDAExecutionProvider.")
    except Exception as e:
        print(f"❌ Error loading ONNX model with CUDA: {e}")

    return str(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX")
    parser.add_argument("--model", type=str, default="yolov8l.pt", help="Path to YOLO model")
    parser.add_argument("--output-dir", type=str, default="./exported_models", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0, cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")

    args = parser.parse_args()
    export_to_onnx(args.model, args.output_dir, args.device, args.fp16)