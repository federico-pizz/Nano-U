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

from src.nas import NASSearcher
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
        print(f"  📦 Copied {keras_path.name} → {model_dest}")

    # Quantize
    print(f"  ⚙️  Quantizing {model_name} → INT8 TFLite...")
    quant_ok = quantize_model(str(model_dest), str(tflite_path))
    quant_result = {
        "status": "success" if quant_ok else "failed",
        "tflite_path": str(tflite_path) if quant_ok else None,
        "size_kb": round(tflite_path.stat().st_size / 1024, 2) if quant_ok and tflite_path.exists() else None,
    }
    if quant_ok:
        print(f"  ✅ Quantization complete: {tflite_path} ({quant_result['size_kb']} KB)")
    else:
        print(f"  ❌ Quantization failed for {model_dest}")

    # Benchmark
    print(f"  📏 Benchmarking {model_name}...")
    bench_result = run_benchmarks(str(model_dest))
    inf = bench_result.get("inference", {})
    print(f"  ✅ Benchmark: {inf.get('avg_latency_ms', '?'):.2f} ms/frame | "
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


# ---------------------------------------------------------------------------
# Evolutionary NAS search
# ---------------------------------------------------------------------------

def run_nas_search(
    config_path: str = "config/experiments.yaml",
    output_dir: str = "results/nas_search/",
    model_name: str = "nano_u",
) -> Dict[str, Any]:
    """Run evolutionary NAS search to find the best architecture.

    Args:
        config_path: Path to experiments YAML.
        output_dir: Where to store per-generation CSVs and best_arch.json.
        model_name: 'nano_u' or 'bu_net'.

    Returns:
        Dict with best_arch, best_fitness, and full generation history.
    """
    full_config = load_config(config_path)
    data_cfg = full_config.get("data", {})
    training_cfg = full_config.get("training", {}).get("common", {})
    nas_cfg = full_config.get("training", {}).get("nas", {})
    model_cfg = full_config.get("models", {}).get(model_name, {})

    # Select model builder
    if model_name == "bu_net":
        from src.models.builders import create_searchable_bu_net
        model_fn: Callable = create_searchable_bu_net
        filters = model_cfg.get("filters", [32, 64, 128])
        bottleneck = model_cfg.get("bottleneck", 256)
        # BU-Net: arch_len = 2*N+1 stages (N pads to 5 internally)
        import math
        padded = max(5, len(filters))
        arch_len = 2 * padded + 1
    else:
        from src.models.builders import create_searchable_nano_u
        model_fn: Callable = create_searchable_nano_u
        filters = model_cfg.get("filters", [16, 32, 64])
        bottleneck = model_cfg.get("bottleneck", 64)
        arch_len = 7  # [enc1,enc2,enc3,bottn,dec1,dec2,dec3]

    searcher = NASSearcher(
        input_shape=tuple(data_cfg.get("input_shape", [48, 64, 3])),
        filters=filters,
        bottleneck=bottleneck,
        population_size=nas_cfg.get("population_size", 4),
        generations=nas_cfg.get("generations", 3),
        arch_len=arch_len,
        model_fn=model_fn,
        output_dir=output_dir,
    )

    def train_proxy(model, epochs):
        from src.data import make_dataset
        processed = full_config.get("data", {}).get("paths", {}).get("processed", {})
        train_cfg = processed.get("train", {})
        val_cfg = processed.get("val", {})

        train_imgs = sorted(Path(train_cfg["img"]).glob("*.png"))
        train_masks = sorted(Path(train_cfg["mask"]).glob("*.png"))
        val_imgs = sorted(Path(val_cfg["img"]).glob("*.png"))
        val_masks = sorted(Path(val_cfg["mask"]).glob("*.png"))

        bs = training_cfg.get("batch_size", 16)
        lr = training_cfg.get("learning_rate", 0.001)
        train_ds = make_dataset([str(f) for f in train_imgs], [str(f) for f in train_masks], batch_size=bs)
        val_ds = make_dataset([str(f) for f in val_imgs], [str(f) for f in val_masks], batch_size=bs, shuffle=False)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)

    print("\n" + "=" * 55)
    print(f"🧬 EVOLUTIONARY NAS SEARCH — model: {model_name}")
    print("=" * 55)

    results = searcher.search(train_proxy)

    print(f"\n🏆 Best Architecture: {results['best_arch']}  "
          f"(fitness={results['best_fitness']:.4f})")

    best_arch_path = Path(output_dir) / "best_arch.json"
    with open(best_arch_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Saved to {best_arch_path}")

    return results
