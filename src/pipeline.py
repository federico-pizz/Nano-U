"""Streamlined training pipeline: automated sweeps and result management."""

import json
import traceback
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import yaml
import tensorflow as tf

from src.train import train_model
from src.utils.config import load_config
from src.benchmarks import run_benchmarks
import shutil

def run_training_pipeline(config_name: str, config_path: str = "config/experiments.yaml",
                        output_dir: str = "results/") -> Dict[str, Any]:
    """Run training pipeline with clean result management."""
    try:
        # Load full configuration and resolve experiment
        full_config = load_config(config_path)
        
        # simplified output directory: results/[config_name]/
        pipeline_dir = Path(output_dir) / config_name
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Run training
        result = train_model(config_path=config_path, experiment_name=config_name, output_dir=str(pipeline_dir))
        
        # The model is now saved as [model_name].keras by train_model
        model_name = result.get("model_name", "model")
        final_model_path = pipeline_dir / f"{model_name}.keras"
        
        if not final_model_path.exists():
            # Fallback check for best_model.keras in case of unexpected save
            best_model_path = pipeline_dir / "best_model.keras"
            if best_model_path.exists():
                best_model_path.rename(final_model_path)
        
        # Save results summary
        results_path = pipeline_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)
            
        return {
            "status": "success",
            "config_name": config_name,
            "pipeline_dir": str(pipeline_dir),
            "results_path": str(results_path),
            **result
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'config_name': config_name,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

def run_pipeline_sweep(experiment_configs: List[str], 
                        config_path: str = "config/experiments.yaml",
                        output_dir: str = "results/sweeps/") -> List[Dict[str, Any]]:
    """Run multiple pipelines with automatic failure handling."""
    results = []
    sweep_dir = Path(output_dir) / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    for config_name in experiment_configs:
        print(f"\n🚀 Running pipeline: {config_name}")
        result = run_training_pipeline(config_name, config_path, str(sweep_dir))
        results.append(result)
        
        # Early stopping on high failure rates
        failed_count = sum(1 for r in results if r['status'] == 'failed')
        if len(results) >= 3 and failed_count > len(results) * 0.5:
            print(f"\n⚠️  Stopping sweep due to high failure rate: {failed_count}/{len(results)}")
            break
            
        # Cooldown to avoid thermal throttling
        if config_name != experiment_configs[-1]:
            time.sleep(1)
            
    # Save sweep summary
    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results

def run_end_to_end_pipeline(config_path: str = "config/experiments.yaml",
                           output_base: str = "results/") -> Dict[str, Any]:
    """Execute full automated pipeline: Teacher -> Student -> Distillation -> Benchmark."""
    print("\n" + "="*50)
    print("🚀 STARTING END-TO-END NANO-U PIPELINE")
    print("="*50)
    
    try:
        # 1. Train Teacher (BU-Net) with NAS
        print("\n--- PHASE 1: Training Teacher (BU-Net) ---")
        teacher_res = run_training_pipeline("bu_net_nas", config_path, output_base)
        if teacher_res["status"] != "success":
            raise RuntimeError(f"Teacher training failed: {teacher_res.get('error')}")
            
        # 2. Automatically prepare weights for Distillation
        print("\n--- PHASE 2: Preparing Teacher Weights ---")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        teacher_keras = Path(output_base) / "bu_net_nas" / "bu_net.keras"
        models_weight_path = models_dir / "bu_net.keras"
        
        if teacher_keras.exists():
            shutil.copy(teacher_keras, models_weight_path)
            print(f"✅ Success: Copied teacher weights to {models_weight_path}")
        else:
            raise FileNotFoundError(f"Could not find trained teacher model at {teacher_keras}")
            
        # 3. Train Student (Nano-U) with Distillation & NAS
        print("\n--- PHASE 3: Training Student (Nano-U) with Distillation ---")
        student_res = run_training_pipeline("distillation_nas", config_path, output_base)
        if student_res["status"] != "success":
            raise RuntimeError(f"Student distillation failed: {student_res.get('error')}")
            
        # 4. Quantize and Benchmark
        print("\n--- PHASE 4: Benchmarking and Quantization ---")
        student_keras = Path(output_base) / "distillation_nas" / "nano_u.keras"
        
        if student_keras.exists():
            benchmark_res = run_benchmarks(str(student_keras))
            print(f"✅ Success: Benchmarking complete. TFLite model generated.")
        else:
            raise FileNotFoundError(f"Could not find trained student model at {student_keras}")
            
        print("\n" + "="*50)
        print("✨ FULL PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return {
            "status": "success",
            "teacher": teacher_res,
            "student": student_res,
            "benchmark": benchmark_res
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
