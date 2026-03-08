"""Performance benchmarking tools for Nano-U models."""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Allow running the script directly (python src/benchmarks.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.data import make_dataset
import argparse

# Import the input shape from the config file globally
_GLOBAL_CONFIG = load_config("config/config.yaml")
INPUT_SHAPE = tuple(_GLOBAL_CONFIG["data"]["input_shape"])

def benchmark_keras_inference(model_path: str,
                              dataset: Optional[tf.data.Dataset] = None,
                              input_shape: Tuple[int, int, int] = INPUT_SHAPE,
                              num_iterations: int = 100) -> Dict[str, Any]:
    """Measure inference speed (latency and throughput) on a standard Keras model."""
    
    print(f"Loading Keras model: {model_path}")
    from src.utils.metrics import BinaryIoU
    with tfmot.quantization.keras.quantize_scope():
        with keras.utils.custom_object_scope({'BinaryIoU': BinaryIoU}):
            model = keras.models.load_model(model_path, compile=False)
    
    if dataset is not None:
        print(f"Benchmarking Keras model {Path(model_path).name} over provided test dataset...")
        
        # Unbatch the dataset to process 1 image at a time
        unbatched_ds = dataset.unbatch()
        dataset_iterator = unbatched_ds.as_numpy_iterator()
        
        # Warmup
        for _ in range(10):
            try:
                x, _ = next(dataset_iterator)
                x_batch = tf.expand_dims(x, 0)
                model(x_batch, training=False)
            except StopIteration:
                break
                
        # Benchmark
        total_time = 0.0
        count = 0
        latencies = []
        
        for x, _ in dataset_iterator:
            x_batch = tf.expand_dims(x, 0)
            
            start_time = time.perf_counter()
            model(x_batch, training=False)
            end_time = time.perf_counter()
            
            latency_s = end_time - start_time
            total_time += latency_s
            latencies.append(latency_s * 1000)
            count += 1
            
        if count == 0:
            print("Warning: Dataset was empty (or consumed in warmup), falling back to dummy data.")
            return benchmark_keras_inference(model_path, dataset=None, input_shape=input_shape, num_iterations=num_iterations)
            
        avg_latency = (total_time / count) * 1000  # ms
        p99_latency = float(np.percentile(latencies, 99)) if latencies else avg_latency
        throughput = count / total_time  # samples/sec
        
        return {
            "avg_latency_ms": avg_latency,
            "p99_latency_ms": p99_latency,
            "throughput_fps": throughput,
            "total_time_sec": total_time,
            "iterations": count
        }
        
    # Fallback to dummy input if no dataset provided
    print(f"Benchmarking Keras with dummy data of shape {input_shape}...")
    x_numpy = np.random.random((1, *input_shape)).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        model(x_numpy, training=False)
        
    # Benchmark
    start_time = time.perf_counter()
    latencies = []
    for _ in range(num_iterations):
        t_start = time.perf_counter()
        model(x_numpy, training=False)
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_iterations) * 1000  # ms
    p99_latency = float(np.percentile(latencies, 99)) if latencies else avg_latency
    throughput = num_iterations / total_time  # samples/sec
    
    return {
        "avg_latency_ms": avg_latency,
        "p99_latency_ms": p99_latency,
        "throughput_fps": throughput,
        "total_time_sec": total_time,
        "iterations": num_iterations
    }

def benchmark_tflite_inference(tflite_path: str, 
                               dataset: Optional[tf.data.Dataset] = None,
                               input_shape: Tuple[int, int, int] = INPUT_SHAPE, 
                               num_iterations: int = 100) -> Dict[str, Any]:
    """Measure inference speed (latency and throughput) directly on INT8 TFLite model."""
    
    # Initialize interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    
    # Get quantization parameters
    scale, zero_point = input_details['quantization']
    if scale == 0.0:
        scale, zero_point = 1.0, 0
    
    input_index = input_details['index']
    
    # If a dataset is provided, we benchmark over it
    if dataset is not None:
        print(f"Benchmarking TFLite model {Path(tflite_path).name} over provided test dataset...")
        
        # Unbatch the dataset to process 1 image at a time
        unbatched_ds = dataset.unbatch()
        dataset_iterator = unbatched_ds.as_numpy_iterator()
        
        # Warmup
        for _ in range(10):
            try:
                x, _ = next(dataset_iterator)
                x_batch = tf.expand_dims(x, 0)
                if input_details['dtype'] == np.int8:
                    x_scaled = tf.round(x_batch / scale) + zero_point
                    x_batch = tf.cast(tf.clip_by_value(x_scaled, -128, 127), tf.int8)
                interpreter.set_tensor(input_index, x_batch)
                interpreter.invoke()
            except StopIteration:
                break
            
        # Benchmark
        total_time = 0.0
        count = 0
        latencies = []
        
        for x, _ in dataset_iterator:
            x_batch = tf.expand_dims(x, 0)
            if input_details['dtype'] == np.int8:
                x_scaled = tf.round(x_batch / scale) + zero_point
                x_batch = tf.cast(tf.clip_by_value(x_scaled, -128, 127), tf.int8)
                
            interpreter.set_tensor(input_index, x_batch)
            
            start_time = time.perf_counter()
            interpreter.invoke()
            end_time = time.perf_counter()
            
            latency_s = end_time - start_time
            total_time += latency_s
            latencies.append(latency_s * 1000)
            count += 1
            
        if count == 0:
            print("Warning: Dataset was empty (or consumed in warmup), falling back to dummy data.")
            return benchmark_tflite_inference(tflite_path, dataset=None, input_shape=input_shape, num_iterations=num_iterations)
            
        avg_latency = (total_time / count) * 1000  # ms
        p99_latency = float(np.percentile(latencies, 99)) if latencies else avg_latency
        throughput = count / total_time  # samples/sec
        
        return {
            "avg_latency_ms": avg_latency,
            "p99_latency_ms": p99_latency,
            "throughput_fps": throughput,
            "total_time_sec": total_time,
            "iterations": count
        }
    
    # Fallback to dummy input if no dataset provided
    print(f"Benchmarking TFLite with dummy data of shape {input_shape}...")
    x = np.random.random((1, *input_shape)).astype(np.float32)
    # Quantize dummy
    if input_details['dtype'] == np.int8:
        x_numpy = np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)
    else:
        x_numpy = x.astype(input_details['dtype'])
        
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_index, x_numpy)
        interpreter.invoke()
    
    # Benchmark
    start_time = time.perf_counter()
    latencies = []
    for _ in range(num_iterations):
        t_start = time.perf_counter()
        interpreter.set_tensor(input_index, x_numpy)
        interpreter.invoke()
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)
    end_time = time.perf_counter()
        
    total_time = end_time - start_time
    avg_latency = (total_time / num_iterations) * 1000  # ms
    p99_latency = float(np.percentile(latencies, 99)) if latencies else avg_latency
    throughput = num_iterations / total_time  # samples/sec
        
    return {
        "avg_latency_ms": avg_latency,
        "p99_latency_ms": p99_latency,
        "throughput_fps": throughput,
        "total_time_sec": total_time,
        "iterations": num_iterations
    }

def run_benchmarks(model_name: str, config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Run full benchmark suite on a saved model."""
    try:
        config = load_config(config_path)
        models_dir = Path(config.get("data", {}).get("paths", {}).get("models_dir", "models"))
        
        # Handle model_name as a full path or a name
        input_path = Path(model_name)
        if input_path.exists() and input_path.is_file():
            model_path = input_path
            is_tflite = model_path.suffix.lower() == ".tflite"
        else:
            if model_name == "nano_u":
                model_path = models_dir / f"{model_name}.tflite"
                is_tflite = True
            else:
                model_path = models_dir / f"{model_name}.h5"
                is_tflite = False
                
            if not model_path.exists():
                # Try fallback .keras if .h5 doesnt exist for keras
                if not is_tflite and (models_dir / f"{model_name}.keras").exists():
                    model_path = models_dir / f"{model_name}.keras"
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"Benchmarking model: {model_path}")
        
        actual_shape = INPUT_SHAPE
        
        # Try to load test dataset
        test_ds = None
        try:
            # Select dataset based on model
            if model_name == "nano_u2":
                dataset_cfg = config.get("data", {}).get("paths", {}).get("secondary", {})
                test_cfg = dataset_cfg.get("test", {})
                print(f"Using secondary dataset ('tinyagri') for {model_name}")
            else:
                data_paths = config.get("data", {}).get("paths", {})
                test_cfg = data_paths.get("processed", {}).get("test", {})
                print(f"Using standard dataset ('botanic_garden') for {model_name}")
            
            t_img_dir = Path(test_cfg.get("img", ""))
            t_mask_dir = Path(test_cfg.get("mask", ""))
            
            if t_img_dir.exists() and t_mask_dir.exists():
                test_img_files = sorted([str(f) for f in t_img_dir.glob("*.png")])
                test_mask_files = sorted([str(f) for f in t_mask_dir.glob("*.png")])
                
                if test_img_files and len(test_img_files) == len(test_mask_files):
                    print(f"Found {len(test_img_files)} test pairs for benchmarking.")
                    test_ds = make_dataset(
                        test_img_files, test_mask_files,
                        batch_size=1, augment=False # Batch size 1 for accurate latency
                    )
        except Exception as e:
            print(f"Could not load test dataset for benchmarking: {e}")
            
        if is_tflite:
            inf_metrics = benchmark_tflite_inference(str(model_path), dataset=test_ds, input_shape=actual_shape)
        else:
            inf_metrics = benchmark_keras_inference(str(model_path), dataset=test_ds, input_shape=actual_shape)
        
        return {
            "inference": inf_metrics,
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Nano-U or BU-Net model inference speed')
    parser.add_argument('model', choices=['bu_net', 'nano_u', 'nano_u2'], help='Model to benchmark')
    args = parser.parse_args()

    print(f"{'='*55}")
    print(f"BENCHMARKING MODEL: {args.model}")
    print(f"{'='*55}")
    
    results = run_benchmarks(args.model)
    if "error" in results:
        print(f"\nBenchmark Failed: {results['error']}")
        print(results.get("traceback", ""))
    else:
        inf = results.get("inference", {})
        print(f"\nFinal Metrics:")
        print(f"  Avg Latency:    {inf.get('avg_latency_ms', '?'):.2f} ms")
        print(f"  P99 Latency:    {inf.get('p99_latency_ms', '?'):.2f} ms")
        print(f"  Throughput:     {inf.get('throughput_fps', '?'):.1f} FPS")
        print(f"  Total Time:     {inf.get('total_time_sec', '?'):.2f} sec")
        print(f"{'='*55}")
