#!/usr/bin/env python3
"""
Simple benchmark script to compare performance between TensorRT and PyTorch YOLO models.
"""
import argparse
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def benchmark_pytorch(model_path, source, num_runs=20, conf=0.25):
    """Benchmark PyTorch YOLO model inference."""
    print(f"\n🚀 Benchmarking PyTorch model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Warmup
    print("Warming up...")
    _ = model(source, conf=conf, verbose=False)
    
    # Benchmark
    print(f"Running {num_runs} inference passes...")
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = model(source, conf=conf, verbose=False)
    
    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    fps = 1 / avg_time
    
    print(f"📊 PyTorch Results:")
    print(f"➡️ Total time: {total_time:.2f} seconds")
    print(f"➡️ Average inference time: {avg_time * 1000:.2f} ms per image")
    print(f"➡️ Throughput: {fps:.2f} FPS")
    
    return {"model": "PyTorch", "avg_time": avg_time, "fps": fps}

def benchmark_tensorrt(engine_path, source, num_runs=20, conf=0.25):
    """Benchmark TensorRT YOLO model inference."""
    print(f"\n🚀 Benchmarking TensorRT model: {engine_path}")
    
    # Use YOLO's built-in engine support
    # For YOLOv8, the engine file can be loaded directly
    model = YOLO(engine_path)
    
    # Warmup
    print("Warming up...")
    _ = model(source, conf=conf, verbose=False)
    
    # Benchmark
    print(f"Running {num_runs} inference passes...")
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = model(source, conf=conf, verbose=False)
    
    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    fps = 1 / avg_time
    
    print(f"📊 TensorRT Results:")
    print(f"➡️ Total time: {total_time:.2f} seconds")
    print(f"➡️ Average inference time: {avg_time * 1000:.2f} ms per image")
    print(f"➡️ Throughput: {fps:.2f} FPS")
    
    return {"model": "TensorRT", "avg_time": avg_time, "fps": fps}

def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO models")
    parser.add_argument("--source", type=str, required=True, help="Path to image or video")
    parser.add_argument("--pytorch", type=str, default="yolov8l.pt", help="Path to PyTorch model")
    parser.add_argument("--tensorrt", type=str, default="/tmp/tensorrt/yolov8l.engine", help="Path to TensorRT engine")
    parser.add_argument("--runs", type=int, default=20, help="Number of inference runs")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    # Print system info
    print("\n🖥️ System Information:")
    print(f"➡️ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"➡️ CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Benchmark each model
    results = []
    
    # Benchmark TensorRT
    tensorrt_results = benchmark_tensorrt(args.tensorrt, args.source, args.runs, args.conf)
    results.append(tensorrt_results)
    
    # Benchmark PyTorch
    pytorch_results = benchmark_pytorch(args.pytorch, args.source, args.runs, args.conf)
    results.append(pytorch_results)
    
    # Compare results
    print("\n🏆 Performance Comparison:")
    fastest = max(results, key=lambda x: x["fps"])
    print(f"Fastest model: {fastest['model']} with {fastest['fps']:.2f} FPS")
    
    # Calculate speedup
    if tensorrt_results and pytorch_results:
        speedup = tensorrt_results["fps"] / pytorch_results["fps"]
        print(f"⚡ TensorRT Speedup: {speedup:.2f}x compared to PyTorch")

if __name__ == "__main__":
    main()