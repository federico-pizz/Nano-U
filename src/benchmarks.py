"""Performance benchmarking tools for Nano-U models."""

import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

def benchmark_inference(model: tf.keras.Model, input_shape: Tuple[int, int, int] = (48, 64, 3), 
                        num_iterations: int = 100) -> Dict[str, float]:
    """Measure inference speed (latency and throughput)."""
    # Create dummy input
    x = np.random.random((1, *input_shape)).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = model(x, training=False)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(x, training=False)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_iterations) * 1000  # ms
    throughput = num_iterations / total_time  # samples/sec
    
    return {
        "avg_latency_ms": avg_latency,
        "throughput_fps": throughput,
        "total_time_sec": total_time,
        "iterations": num_iterations
    }

def validate_tflite_optimization(model: tf.keras.Model, 
                                 representative_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Convert to TFLite and validate size/optimizations."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if representative_data is not None:
        def representative_data_gen():
            for i in range(len(representative_data)):
                yield [representative_data[i:i+1].astype(np.float32)]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
    try:
        tflite_model = converter.convert()
        model_size_kb = len(tflite_model) / 1024
        
        # Save TFLite model next to original if possible
        tflite_path = None
        if hasattr(model, 'output_path') and model.output_path:
            tflite_path = Path(model.output_path).parent / "model.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
        
        return {
            "status": "success",
            "size_kb": model_size_kb,
            "is_quantized": representative_data is not None,
            "tflite_path": str(tflite_path) if tflite_path else None
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

def run_benchmarks(model_path: str, input_shape: Tuple[int, int, int] = (48, 64, 3)) -> Dict[str, Any]:
    """Run full benchmark suite on a saved model."""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        
        print(f"📏 Benchmarking model: {model_path}")
        
        # Attach path for TFLite saving
        model.output_path = model_path
        
        inf_metrics = benchmark_inference(model, input_shape)
        tflite_metrics = validate_tflite_optimization(model)
        
        return {
            "inference": inf_metrics,
            "tflite": tflite_metrics,
            "parameters": model.count_params()
        }
    except Exception as e:
        return {"error": str(e)}

class MemoryProfilingCallback(tf.keras.callbacks.Callback):
    """Monitor system and GPU memory usage during training."""
    
    def __init__(self):
        super().__init__()
        self.history = []
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            self.process = None

    def on_epoch_end(self, epoch, logs=None):
        metrics = {"epoch": epoch}
        
        # System memory
        if self.process:
            mem_info = self.process.memory_info()
            metrics["sys_rss_mb"] = mem_info.rss / (1024 * 1024)
            
        # GPU memory (if available)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Basic GPU memory info via TF
                # Note: TF doesn't provide easy direct MB usage for current process
                # but we can log that we are using GPU
                metrics["gpu_count"] = len(gpus)
            except Exception:
                pass
                
        self.history.append(metrics)
        print(f" 🧠 Memory Usage (RSS): {metrics.get('sys_rss_mb', 0):.2f} MB")

    def get_summary(self) -> Dict[str, float]:
        if not self.history:
            return {}
        rss_values = [h.get("sys_rss_mb", 0) for h in self.history]
        return {
            "max_rss_mb": float(np.max(rss_values)),
            "avg_rss_mb": float(np.mean(rss_values))
        }
