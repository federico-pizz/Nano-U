"""
Comprehensive tests for Nano-U model implementations.

Tests model creation, serialization, parameter counting, and functional correctness.
"""

import pytest
import tensorflow as tf
from tensorflow.keras import Model
from typing import Tuple, List, Dict, Any

from src.models import (
    create_nano_u,
    create_bu_net,
    create_nano_u_functional,
    create_bu_net_functional,
    create_model_from_config,
    get_model_summary,
    count_parameters,
    validate_model_serialization,
    get_model_config
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def nano_u_model() -> Model:
    """Create a NanoU model instance for testing."""
    return create_nano_u(
        input_shape=(48, 64, 3),
        filters=[16, 32],
        bottleneck=64,
        name='test_nano_u'
    )


@pytest.fixture
def bu_net_model() -> Model:
    """Create a BU_Net model instance for testing."""
    return create_bu_net(
        input_shape=(48, 64, 3),
        filters=[32, 64, 128],
        bottleneck=256,
        name='test_bu_net'
    )


@pytest.fixture
def config() -> Dict[str, Any]:
    """Create a model configuration dictionary."""
    return {
        'model_name': 'nano_u',
        'input_shape': [48, 64, 3],
        'filters': [16, 32],
        'bottleneck': 64
    }


@pytest.fixture
def input_data() -> tf.Tensor:
    """Create synthetic input data for model testing."""
    return tf.random.normal((1, 48, 64, 3))


# =============================================================================
# Model Creation Tests
# =============================================================================


def test_nano_u_creation(nano_u_model: Model):
    """Test NanoU model creation and basic properties."""
    assert nano_u_model is not None
    assert nano_u_model.name == 'test_nano_u'
    assert nano_u_model.input_shape == (None, 48, 64, 3)
    assert nano_u_model.output_shape == (None, 48, 64, 1)
    assert len(nano_u_model.layers) > 0



def test_bu_net_creation(bu_net_model: Model):
    """Test BU_Net model creation and basic properties."""
    assert bu_net_model is not None
    assert bu_net_model.name == 'test_bu_net'
    assert bu_net_model.input_shape == (None, 48, 64, 3)
    assert bu_net_model.output_shape == (None, 48, 64, 1)
    assert len(bu_net_model.layers) > 0



def test_model_from_config(config: Dict[str, Any], nano_u_model: Model):
    """Test model creation from configuration dictionary."""
    model = create_model_from_config(config)
    assert model is not None
    assert model.name == 'nano_u'
    assert model.input_shape == (None, 48, 64, 3)



def test_functional_api_compatibility():
    """Test that functional API versions match the main implementations."""
    # Test NanoU functional API
    functional_nano_u = create_nano_u_functional()
    regular_nano_u = create_nano_u()
    
    assert functional_nano_u.input_shape == regular_nano_u.input_shape
    assert functional_nano_u.output_shape == regular_nano_u.output_shape
    
    # Test BU_Net functional API
    functional_bu_net = create_bu_net_functional()
    regular_bu_net = create_bu_net()
    
    assert functional_bu_net.input_shape == regular_bu_net.input_shape
    assert functional_bu_net.output_shape == regular_bu_net.output_shape


# =============================================================================
# Parameter Counting Tests
# =============================================================================


def test_parameter_counting(nano_u_model: Model):
    """Test parameter counting functionality."""
    params = count_parameters(nano_u_model)
    assert params > 0
    assert isinstance(params, int)
    
    # Test that parameter count is reasonable for NanoU
    assert params < 50000  # Should be very lightweight



def test_parameter_counting_bu_net(bu_net_model: Model):
    """Test parameter counting for BU_Net model."""
    params = count_parameters(bu_net_model)
    assert params > 0
    assert isinstance(params, int)
    
    # Test that BU_Net has more parameters than NanoU
    nano_u_params = count_parameters(create_nano_u())
    assert params > nano_u_params


# =============================================================================
# Model Summary Tests
# =============================================================================


def test_model_summary(nano_u_model: Model):
    """Test model summary generation."""
    summary = get_model_summary(nano_u_model)
    assert isinstance(summary, str)
    assert len(summary) > 0
    
    # Check that summary contains expected information
    assert 'test_nano_u' in summary
    assert 'Total params' in summary



def test_model_summary_bu_net(bu_net_model: Model):
    """Test model summary generation for BU_Net."""
    summary = get_model_summary(bu_net_model)
    assert isinstance(summary, str)
    assert len(summary) > 0
    
    # Check that summary contains expected information
    assert 'test_bu_net' in summary
    assert 'Total params' in summary


# =============================================================================
# Model Configuration Tests
# =============================================================================


def test_model_config(nano_u_model: Model):
    """Test model configuration extraction."""
    config = get_model_config(nano_u_model)
    assert isinstance(config, dict)
    assert 'model_name' in config
    assert 'input_shape' in config
    assert 'parameter_count' in config
    assert 'layer_count' in config
    
    assert config['model_name'] == 'test_nano_u'
    assert config['input_shape'] == [48, 64, 3]



def test_model_config_bu_net(bu_net_model: Model):
    """Test model configuration extraction for BU_Net."""
    config = get_model_config(bu_net_model)
    assert isinstance(config, dict)
    assert 'model_name' in config
    assert 'input_shape' in config
    assert 'parameter_count' in config
    assert 'layer_count' in config
    
    assert config['model_name'] == 'test_bu_net'
    assert config['input_shape'] == [48, 64, 3]


# =============================================================================
# Model Serialization Tests
# =============================================================================


def test_model_serialization(nano_u_model: Model):
    """Test model serialization and deserialization."""
    assert validate_model_serialization(nano_u_model)



def test_model_serialization_bu_net(bu_net_model: Model):
    """Test model serialization for BU_Net."""
    assert validate_model_serialization(bu_net_model)



def test_model_save_load(nano_u_model: Model):
    """Test model save and load functionality."""
    try:
        # Save model
        model_path = 'temp_model_save.keras'
        nano_u_model.save(model_path)
        
        # Load model
        loaded_model = tf.keras.models.load_model(model_path)
        
        # Verify model integrity
        assert loaded_model.input_shape == nano_u_model.input_shape
        assert loaded_model.output_shape == nano_u_model.output_shape
        assert loaded_model.name == nano_u_model.name
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(model_path, ignore_errors=True)


# =============================================================================
# Model Inference Tests
# =============================================================================


def test_model_inference(nano_u_model: Model, input_data: tf.Tensor):
    """Test model inference with synthetic data."""
    output = nano_u_model(input_data, training=False)
    assert output is not None
    assert output.shape == (1, 48, 64, 1)
    
    # Check if last layer has sigmoid activation
    # In some models (like those for microflow), sigmoid is omitted for compatibility
    last_layer = nano_u_model.layers[-1]
    has_sigmoid = hasattr(last_layer, 'activation') and last_layer.activation == tf.keras.activations.sigmoid
    
    if has_sigmoid:
        assert tf.reduce_max(output).numpy() <= 1.0
        assert tf.reduce_min(output).numpy() >= 0.0


def test_model_inference_bu_net(bu_net_model: Model, input_data: tf.Tensor):
    """Test BU_Net inference with synthetic data."""
    output = bu_net_model(input_data, training=False)
    assert output is not None
    assert output.shape == (1, 48, 64, 1)
    
    # Check if last layer has sigmoid activation
    last_layer = bu_net_model.layers[-1]
    has_sigmoid = hasattr(last_layer, 'activation') and last_layer.activation == tf.keras.activations.sigmoid
    
    if has_sigmoid:
        assert tf.reduce_max(output).numpy() <= 1.0
        assert tf.reduce_min(output).numpy() >= 0.0


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_invalid_model_name():
    """Test error handling for invalid model names."""
    with pytest.raises(ValueError):
        create_model_from_config({'model_name': 'invalid_model'})



def test_empty_config():
    """Test error handling for empty configuration."""
    with pytest.raises(ValueError):
        create_model_from_config({})



def test_minimal_config():
    """Test model creation with minimal configuration."""
    model = create_model_from_config({'model_name': 'nano_u'})
    assert model is not None
    assert model.input_shape == (None, 48, 64, 3)


# =============================================================================
# Performance Tests
# =============================================================================


def test_model_creation_speed():
    """Test that model creation is reasonably fast."""
    import time
    
    start_time = time.time()
    create_nano_u()
    creation_time = time.time() - start_time
    
    assert creation_time < 1.0  # Should be very fast



def test_model_size(nano_u_model: Model):
    """Test that model size is reasonable for microcontroller deployment."""
    # Test that model is lightweight
    params = count_parameters(nano_u_model)
    assert params < 50000  # Should be very lightweight for microcontroller



def test_bu_net_size(bu_net_model: Model):
    """Test that BU_Net size is reasonable for teacher model."""
    # Test that BU_Net is larger than NanoU but still reasonable
    params = count_parameters(bu_net_model)
    nano_u_params = count_parameters(create_nano_u())
    
    assert params > nano_u_params  # BU_Net should be larger
    assert params < 5000000  # Should still be reasonable for training


# =============================================================================
# Configuration Tests
# =============================================================================


def test_config_validation():
    """Test that model configurations are properly validated."""
    # Test valid configurations
    valid_configs = [
        {'model_name': 'nano_u', 'input_shape': [48, 64, 3]},
        {'model_name': 'bu_net', 'input_shape': [48, 64, 3]},
        {'model_name': 'nano_u', 'filters': [16, 32], 'bottleneck': 64}
    ]
    
    for config in valid_configs:
        try:
            create_model_from_config(config)
        except Exception as e:
            pytest.fail(f'Valid config {config} failed: {e}')



def test_config_defaults():
    """Test that default values are properly applied."""
    # Test with minimal config
    model = create_model_from_config({'model_name': 'nano_u'})
    assert model is not None
    assert model.input_shape == (None, 48, 64, 3)
    
    # Test with partial config
    model = create_model_from_config({
        'model_name': 'bu_net',
        'input_shape': [64, 80, 3]
    })
    assert model is not None
    assert model.input_shape == (None, 64, 80, 3)


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_pipeline():
    """Test full model creation and validation pipeline."""
    # Create model
    model = create_nano_u()
    
    # Validate serialization
    assert validate_model_serialization(model)
    
    # Get configuration
    config = get_model_config(model)
    assert isinstance(config, dict)
    
    # Test inference
    input_data = tf.random.normal((1, 48, 64, 3))
    output = model(input_data, training=False)
    assert output.shape == (1, 48, 64, 1)
    
    # Test parameter counting
    params = count_parameters(model)
    assert params > 0


if __name__ == '__main__':
    # Run tests when executed directly
    pytest.main([__file__, '-vv'])