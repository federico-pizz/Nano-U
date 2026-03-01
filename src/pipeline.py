"""Core pipeline functions: single training run, NAS search, quantize+benchmark helper."""

import json
import shutil
import traceback
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
import tensorflow as tf

from src.train import train_model
from src.utils.config import load_config
from src.benchmarks import run_benchmarks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quantize_and_benchmark(
    keras_path: Union[str, Path],
    models_dir: Union[str, Path] = "models/",
) -> Dict[str, Any]:
    """Quantize a model to INT8 TFLite and benchmark inference speed."""
    from src.quantize_model import quantize_model

    keras_path = Path(keras_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = keras_path.stem
    model_dest = models_dir / keras_path.name
    tflite_path = models_dir / f"{model_name}.tflite"

    # Copy .keras to models/
    if keras_path != model_dest:
        shutil.copy(keras_path, model_dest)
        print(f"Copied {keras_path.name} → {model_dest}")

    # Quantize
    print(f"Quantizing {model_name} → INT8 TFLite...")
    quant_ok = quantize_model(str(model_dest), str(tflite_path))
    quant_result = {
        "status": "success" if quant_ok else "failed",
        "tflite_path": str(tflite_path) if quant_ok else None,
        "size_kb": round(tflite_path.stat().st_size / 1024, 2) if quant_ok and tflite_path.exists() else None,
    }
    if quant_ok:
        print(f"Quantization complete: {tflite_path} ({quant_result['size_kb']} KB)")
    else:
        print(f"Quantization failed for {model_dest}")

    # Benchmark
    print(f"Benchmarking {model_name}...")
    bench_result = run_benchmarks(str(model_dest))
    inf = bench_result.get("inference", {})
    print(f"Benchmark: {inf.get('avg_latency_ms', '?'):.2f} ms/frame | "
          f"{inf.get('throughput_fps', '?'):.1f} FPS | "
          f"{bench_result.get('parameters', '?'):,} params")

    return {"quantization": quant_result, "benchmark": bench_result}


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_training_pipeline(
    config_name: str,
    config_path: str = "config/experiments.yaml",
    output_dir: str = "results/",
) -> Dict[str, Any]:
    """Train one named experiment and return result dict."""
    try:
        pipeline_dir = Path(output_dir) / config_name
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        result = train_model(
            config_path=config_path,
            experiment_name=config_name,
            output_dir=str(pipeline_dir)
        )

        model_name = result.get("model_name", "model")
        final_model_path = pipeline_dir / f"{model_name}.keras"

        # Fallback: rename temp_model.keras if train_model saved it that way
        if not final_model_path.exists():
            temp_model = Path("models/temp_model.keras")
            if temp_model.exists():
                shutil.move(str(temp_model), str(final_model_path))

        results_path = pipeline_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)

        return {
            "status": "success",
            "config_name": config_name,
            "pipeline_dir": str(pipeline_dir),
            "model_path": str(final_model_path),
            "results_path": str(results_path),
            **result,
        }

    except Exception as e:
        return {
            "status": "failed",
            "config_name": config_name,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }