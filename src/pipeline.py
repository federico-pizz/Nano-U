"""Core pipeline functions: single training run, NAS search, quantize+benchmark helper."""

import os
import sys
import json
import shutil
import traceback
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
import tensorflow as tf
import numpy as np

# Allow running the script directly (python src/pipeline.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import train_model
from src.utils import NumpyEncoder
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_best_epoch_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse a per-epoch history dict to a single-value summary.

    Val metrics are taken at the epoch with the lowest val_loss (the saved
    checkpoint epoch). Train metrics are taken at the final epoch.
    """
    val_losses = raw.get("val_loss")
    best_epoch = (
        int(np.argmin(val_losses)) if isinstance(val_losses, list) and val_losses else None
    )
    out = {}
    for k, v in raw.items():
        if isinstance(v, list) and len(v) > 0:
            out[k] = v[best_epoch] if best_epoch is not None and k.startswith("val_") else v[-1]
        else:
            out[k] = v
    return out


def export_int8(
    keras_path: Union[str, Path],
    models_dir: Union[str, Path] = "models/",
) -> Dict[str, Any]:
    """Convert a Keras model to a calibrated INT8 TFLite file.

    Also produces a companion _quant_params.json consumed by firmware/build.rs
    at compile time to embed correct scale/zero-point values in the binary.
    """
    from src.quantize_model import quantize_model

    keras_path = Path(keras_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = keras_path.stem
    model_dest = models_dir / keras_path.name
    tflite_path = models_dir / f"{model_name}.tflite"
    quant_params_path = models_dir / f"{model_name}_quant_params.json"

    if keras_path != model_dest:
        shutil.copy(keras_path, model_dest)
        print(f"Copied {keras_path.name} → {model_dest}")

    print(f"Quantizing {model_name} → INT8 TFLite...")
    quant_ok = quantize_model(str(model_dest), str(tflite_path))
    quant_result = {
        "status": "success" if quant_ok else "failed",
        "tflite_path": str(tflite_path) if quant_ok else None,
        "quant_params_path": str(quant_params_path) if quant_ok and quant_params_path.exists() else None,
        "size_kb": round(tflite_path.stat().st_size / 1024, 2) if quant_ok and tflite_path.exists() else None,
    }
    if quant_ok:
        print(f"TFLite export complete: {tflite_path} ({quant_result['size_kb']} KB)")
        print(f"Quant params saved:     {quant_params_path}")
    else:
        print(f"Quantization failed for {model_dest}")

    return quant_result


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_training(
    config_name: str,
    config_path: str = "config/experiments.yaml",
    output_dir: str = "results/",
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one named experiment and return result dict.

    config_overrides: merged on top of the YAML config before training begins.
        Use this to inject dynamic values (e.g. teacher_weights resolved at runtime).
    """
    try:
        pipeline_dir = Path(output_dir) / config_name
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        result = train_model(
            config_path=config_path,
            experiment_name=config_name,
            output_dir=str(pipeline_dir),
            extra_config=config_overrides,
        )

        results_path = pipeline_dir / "results.json"
        
        summary_result = result.copy()
        if "final_metrics" in summary_result and isinstance(summary_result["final_metrics"], dict):
            summary_result["final_metrics"] = _select_best_epoch_metrics(summary_result["final_metrics"])
            
        with open(results_path, "w") as f:
            json.dump(summary_result, f, indent=2, cls=NumpyEncoder)

        return {
            "status": "success",
            "config_name": config_name,
            "pipeline_dir": str(pipeline_dir),
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