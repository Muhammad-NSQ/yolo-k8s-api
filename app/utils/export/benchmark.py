#!/usr/bin/env python3
"""
Enhanced benchmark script to compare performance between ONNX and PyTorch YOLO models.
Includes improved GPU utilization and monitoring for ONNX models.
"""
import time
import argparse
import cv2
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

def benchmark_pytorch(model_path, image, num_runs=20, warmup=5, device='cuda:0'):
    """Benchmark PyTorch YOLO model inference."""
    print(f"🚀 Loading PyTorch model: {model_path} on {device}")

    try:
        model = YOLO(model_path)
        model.model.to(device)  # Ensure model is on GPU
    except Exception as e:
        print(f"❌ Failed to load model {model_path}: {e}")
        return None

    print("\n=== Benchmarking PyTorch Model ===")

    # Prediction arguments
    pred_args = {'conf': 0.25, 'iou': 0.45, 'max_det': 300, 'verbose': False}

    # Warmup runs
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.predict(image, **pred_args)

    # Benchmark runs using CUDA events for precise GPU timing
    inference_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(num_runs):
        # Synchronize before starting measurement
        torch.cuda.synchronize()
        start_event.record()
        
        with torch.no_grad():
            _ = model.predict(image, **pred_args)
            
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        inference_times.append(elapsed_time)

    # Report GPU memory usage
    current_mem = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"🧠 PyTorch GPU Memory: Current {current_mem:.2f} MB, Peak {peak_mem:.2f} MB")

    return compute_stats(inference_times, "PyTorch")


def benchmark_onnx(model_path, image, num_runs=20, warmup=5):
    """Benchmark ONNX model inference performance with improved GPU utilization."""
    print(f"🚀 Loading ONNX model: {model_path}")
    
    # Create session options for detailed logging
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0  # Verbose logging
    sess_options.log_verbosity_level = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Set execution providers - always try TensorRT first
    providers = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider", 
        "CPUExecutionProvider"
    ]
    
    provider_options = [
        {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "/tmp/tensorrt/cache"
        },
        {
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True
        },
        {}
    ]
    
    # Load ONNX model with GPU providers
    try:
        session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options
        )
        
        # Print which provider is actually being used
        active_provider = session.get_providers()[0]
        print(f"🔥 Using execution provider: {active_provider}")
        
    except Exception as e:
        print(f"❌ Failed to load ONNX model with TensorRT/CUDA: {e}")
        print("⚠️ Falling back to CPU provider")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    # Get model input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    height, width = input_shape[2], input_shape[3]  # Dynamically get ONNX input size
    print(f"📏 ONNX input shape: {input_shape}")

    # Preprocess image dynamically with FP16 precision for TensorRT
    image_resized = cv2.resize(image, (width, height))
    image_transposed = np.transpose(image_resized, (2, 0, 1))
    image_normalized = image_transposed / 255.0
    image_input = np.expand_dims(image_normalized, axis=0).astype(np.float16)

    print(f"🧪 Input data type: {image_input.dtype}")
    
    # Create CUDA events for GPU timing if using GPU provider
    using_gpu = "CUDA" in active_provider or "Tensorrt" in active_provider
    if using_gpu and torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Force CUDA initialization
        dummy = torch.zeros(1, device='cuda')
        
        print("⏱️ Using CUDA events for accurate GPU timing")
    
    # Warmup runs
    print(f"🔄 Performing {warmup} warmup runs...")
    for i in range(warmup):
        _ = session.run(None, {input_name: image_input})
        if i == 0:
            print("✅ First inference completed successfully")

    # Benchmark runs
    print(f"📊 Starting {num_runs} benchmark runs...")
    inference_times = []
    
    for i in range(num_runs):
        if using_gpu and torch.cuda.is_available():
            # GPU timing with CUDA events
            torch.cuda.synchronize()
            start_event.record()
            
            _ = session.run(None, {input_name: image_input})
            
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        else:
            # CPU timing fallback
            start_time = time.time()
            _ = session.run(None, {input_name: image_input})
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            
        inference_times.append(elapsed_time)
        
        # Print progress for long benchmarks
        if (i+1) % 5 == 0:
            print(f"Progress: {i+1}/{num_runs} runs completed")
    
    # Try to report GPU memory usage if possible
    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"🧠 ONNX GPU Memory: Current {current_mem:.2f} MB, Peak {peak_mem:.2f} MB")

    return compute_stats(inference_times, "ONNX")


def compute_stats(inference_times, model_type):
    """Compute and print performance statistics."""
    avg_time = np.mean(inference_times)
    median_time = np.median(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_dev = np.std(inference_times)
    fps = 1000 / avg_time  # Convert ms to FPS

    print(f"\n📊 {model_type} Model Performance:")
    print(f"➡️ Average inference time: {avg_time:.2f} ms")
    print(f"➡️ Median inference time: {median_time:.2f} ms")
    print(f"➡️ Min/Max time: {min_time:.2f}/{max_time:.2f} ms")
    print(f"➡️ Standard deviation: {std_dev:.2f} ms")
    print(f"➡️ Throughput: {fps:.2f} FPS")

    return {
        "model_type": model_type, 
        "avg_time_ms": avg_time, 
        "median_time_ms": median_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_dev_ms": std_dev,
        "fps": fps
    }


def compare_models(image_path, runs=20):
    """Compare performance between ONNX and PyTorch models."""
    # Print system information
    print("\n🖥️ System Information:")
    print(f"➡️ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"➡️ CUDA Version: {torch.version.cuda}")
        print(f"➡️ CUDA Device: {torch.cuda.get_device_name(0)}")
    
    print(f"➡️ ONNX Runtime Version: {ort.__version__}")
    print(f"➡️ Available Providers: {ort.get_available_providers()}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"📷 Loaded image: {image_path}, Shape: {image.shape}")

    results = []
    
    # Benchmark ONNX model
    onnx_model_path = "/tmp/tensorrt/yolov8l_fp16.onnx"
    onnx_results = benchmark_onnx(onnx_model_path, image, runs)
    results.append(onnx_results)
    
    # Benchmark PyTorch model
    pytorch_results = benchmark_pytorch("yolov8l.pt", image, runs)
    if pytorch_results:
        results.append(pytorch_results)
    
    # Compare results
    print("\n🏆 Performance Comparison:")
    fastest_model = min(results, key=lambda x: x["avg_time_ms"])
    print(f"Fastest model: {fastest_model['model_type']} ({fastest_model['fps']:.2f} FPS)")
    
    # Calculate speedups
    baseline = pytorch_results["avg_time_ms"]
    for result in results:
        if result["model_type"] != "PyTorch":
            speedup = baseline / result["avg_time_ms"]
            print(f"⚡ {result['model_type']} Speedup: {speedup:.2f}x compared to PyTorch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YOLO model inference")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--runs", type=int, default=20, help="Number of inference runs")

    args = parser.parse_args()
    compare_models(args.image, args.runs)