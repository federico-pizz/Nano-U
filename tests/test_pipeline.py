"""
Integration tests: full training run, NAS stability, model size.

Run from project root: pytest tests/test_pipeline.py -v
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import tensorflow as tf

# Project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import create_nano_u, create_model_from_config, count_parameters
from src.train import train_model, train_single_model, _get_experiment_config, _get_train_val_data_synthetic
from src.utils.config import load_config
from src.nas import compute_layer_redundancy, validate_nas_computation, NASCallback


def test_model_instantiation():
    """Models build and compile without errors."""
    model = create_nano_u(input_shape=(48, 64, 3))
    assert model is not None
    assert len(model.layers) > 0
    model.compile(optimizer="adam", loss="binary_crossentropy")
    assert model.optimizer is not None


def test_training_pipeline_synthetic():
    """Short training run with synthetic data (2 epochs, small batch)."""
    config = {
        "model_name": "nano_u",
        "input_shape": [48, 64, 3],
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.001,
    }
    train_data, val_data = _get_train_val_data_synthetic(config, num_train=16, num_val=8)
    model = create_model_from_config(config)
    history = train_single_model(model, config, train_data, val_data)
    assert "loss" in history.history
    assert len(history.history["loss"]) == 2


def test_nas_stability():
    """NAS redundancy: positive condition number, score in [0, 1]."""
    activations = tf.random.normal((32, 24, 32, 64))
    r = compute_layer_redundancy(activations)
    assert 0.0 <= r["redundancy_score"] <= 1.0
    assert r["condition_number"] > 0
    assert r["rank"] > 0


def test_nas_validate():
    """NAS validation helper runs without error."""
    assert validate_nas_computation() is True


def test_model_size_constraints():
    """Nano-U stays under 50K params and quantizes to < 200KB."""
    model = create_nano_u(input_shape=(48, 64, 3))
    n = count_parameters(model)
    assert n < 50_000, f"Model too large: {n} parameters"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    assert len(tflite_model) < 200_000, f"Quantized model too large: {len(tflite_model)} bytes"


def test_experiment_config_resolution():
    """Experiment config is resolved from full config (experiments section)."""
    path = os.path.join(os.path.dirname(__file__), "..", "config", "experiments.yaml")
    if not os.path.exists(path):
        pytest.skip("config/experiments.yaml not found")
    full = load_config(path)
    cfg = _get_experiment_config(full, "quick_test")
    assert "epochs" in cfg or "model_name" in cfg


def test_train_model_quick(tmp_path):
    """train_model runs without error for quick_test (uses synthetic data)."""
    path = os.path.join(os.path.dirname(__file__), "..", "config", "experiments.yaml")
    if not os.path.exists(path):
        pytest.skip("config/experiments.yaml not found")
    out = str(tmp_path)
    result = train_model(config_path=path, experiment_name="quick_test", output_dir=out)
    assert result["status"] == "success"
    assert "model_path" in result
