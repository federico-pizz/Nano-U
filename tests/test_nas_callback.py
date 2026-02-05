"""
Unit tests for NASMonitorCallback.

Tests the callback-based NAS monitoring system to ensure:
1. Callback initializes correctly
2. Metrics are computed properly
3. TensorBoard logging works
4. CSV export functions correctly
5. Works with different model architectures
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import gc

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nas import NASCallback as NASMonitorCallback


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean up GPU memory before and after each test."""
    # Clear before test
    tf.keras.backend.clear_session()
    gc.collect()
    yield
    # Clear after test
    tf.keras.backend.clear_session()
    gc.collect()


def create_simple_model(input_shape=(32, 32, 3)):
    """Create a simple functional model for testing."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', name='conv1')(inputs)
    x = tf.keras.layers.Conv2D(8, 3, activation='relu', name='conv2')(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='simple_model')


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return create_simple_model()


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset for testing."""
    # Generate random data
    x_train = np.random.randn(100, 32, 32, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, (100, 1)).astype(np.float32)
    x_val = np.random.randn(20, 32, 32, 3).astype(np.float32)
    y_val = np.random.randint(0, 2, (20, 1)).astype(np.float32)
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(10)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(10)
    
    return train_ds, val_ds


def test_callback_initialization(temp_dir):
    """Test that NASMonitorCallback initializes correctly."""
    log_dir = os.path.join(temp_dir, "logs")
    csv_path = os.path.join(temp_dir, "metrics.csv")
    
    callback = NASMonitorCallback(
            layers_to_monitor=['conv1', 'conv2'],
            log_frequency=1,
            output_dir=temp_dir
        )
    
    assert callback.layers_to_monitor == ['conv1', 'conv2']
    assert callback.log_frequency == 1
    assert callback.output_dir == temp_dir
    assert len(callback.redundancy_history) == 0


def test_callback_with_training(simple_model, dummy_dataset, temp_dir):
    """Test callback during actual training."""
    train_ds, val_ds = dummy_dataset
    log_dir = os.path.join(temp_dir, "logs")
    csv_path = os.path.join(temp_dir, "metrics.csv")
    
    callback = NASMonitorCallback(
            layers_to_monitor=['conv1', 'conv2'],
            log_frequency=1,
            output_dir=temp_dir,
            validation_data=val_ds
        )
    
    simple_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train for 3 epochs
    history = simple_model.fit(
        train_ds,
        epochs=3,
        callbacks=[callback],
        verbose=0
    )
    
    # Check that metrics were recorded
    assert len(callback.redundancy_history) == 3
    
    # Check CSV was created
    assert os.path.exists(csv_path)
    
    # Verify CSV contents
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert 'epoch' in df.columns
    assert 'conv1_redundancy_score' in df.columns
    assert 'conv1_condition_number' in df.columns
    assert 'conv1_rank' in df.columns
    assert 'conv1_num_channels' in df.columns


def test_metrics_computation(simple_model, dummy_dataset, temp_dir):
    """Test that metrics are computed correctly."""
    train_ds, val_ds = dummy_dataset
    csv_path = os.path.join(temp_dir, "metrics.csv")
    
    callback = NASMonitorCallback(
            layers_to_monitor=['conv1', 'conv2'],
            log_frequency=1,
            output_dir=temp_dir,
            validation_data=val_ds
        )
    
    simple_model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    
    simple_model.fit(
        train_ds,
        epochs=2,
        callbacks=[callback],
        verbose=0
    )
    
    # Check metric values are reasonable
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Redundancy should be between 0 and 1
    assert all(0 <= x <= 1 for x in df['conv1_redundancy_score'])
    
    # Condition number should be positive (>=1)
    assert all(x >= 1 for x in df['conv1_condition_number'])


def test_tensorboard_logging(simple_model, dummy_dataset, temp_dir):
    """Test that TensorBoard logging works."""
    train_ds, val_ds = dummy_dataset
    log_dir = os.path.join(temp_dir, "tensorboard")
    
    callback = NASMonitorCallback(
            layers_to_monitor=['conv1', 'conv2'],
            log_frequency=1,
            output_dir=temp_dir,
            log_dir=log_dir,
            validation_data=val_ds
        )
    
    simple_model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    
    simple_model.fit(
        train_ds,
        epochs=2,
        callbacks=[callback],
        verbose=0
    )
    
    # Check TensorBoard logs were created
    assert os.path.exists(log_dir)
    # TensorBoard creates event files
    event_files = list(Path(log_dir).rglob('events.out.tfevents.*'))
    assert len(event_files) > 0


def test_batch_level_monitoring(simple_model, dummy_dataset, temp_dir):
    """Test batch-level monitoring."""
    train_ds, val_ds = dummy_dataset
    
    callback = NASMonitorCallback(
            layers_to_monitor=['conv1', 'conv2'],
            log_frequency=5,  # Monitor every 5 batches
            output_dir=temp_dir,
            monitor_frequency='batch',
            validation_data=val_ds
        )
    
    simple_model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    
    simple_model.fit(
        train_ds,
        epochs=1,
        callbacks=[callback],
        verbose=0
    )
    
    # Dataset has 100 samples, batch size 10 = 10 batches per epoch
    # With log_frequency=5, we get 2 recordings
    assert len(callback.redundancy_history) == 2


def test_no_validation_data(simple_model, temp_dir):
    """Test callback behavior when no validation data is provided."""
    # Create train data only
    x_train = np.random.randn(50, 32, 32, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, (50, 1)).astype(np.float32)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(10)
    
    callback = NASMonitorCallback(
            layers_to_monitor=['conv1', 'conv2'],
            log_frequency=1,
            output_dir=temp_dir
        )
    
    simple_model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    
    # Train without validation data
    simple_model.fit(
        train_ds,
        epochs=2,
        callbacks=[callback],
        verbose=0
    )
    
    # Callback should handle gracefully - no metrics recorded because no val data is attached
    assert len(callback.redundancy_history) == 0


def test_get_metrics_method(simple_model, dummy_dataset, temp_dir):
    """Test the get_metrics() method."""
    train_ds, val_ds = dummy_dataset
    
    callback = NASMonitorCallback(
            layers_to_monitor=['conv1', 'conv2'],
            log_frequency=1,
            output_dir=temp_dir,
            validation_data=val_ds
        )
    
    simple_model.compile(optimizer='adam', loss='binary_crossentropy')
    simple_model.fit(train_ds, epochs=3, callbacks=[callback], verbose=0)
    
    # Get metrics
    metrics = callback.get_metrics()
    
    assert isinstance(metrics, dict)
    assert 'conv1_redundancy_score' in metrics
    assert len(metrics['conv1_redundancy_score']) == 3  # 3 epochs


def test_different_output_shapes(temp_dir):
    """Test callback with different model output shapes."""
    # Test with image segmentation output (4D tensor)
    def create_segmentation_model():
        inputs = tf.keras.layers.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(16, 3, padding='same', name='conv_seg')(inputs)
        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model = create_segmentation_model()
    
    # Create image data
    x_train = np.random.randn(20, 32, 32, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, (20, 32, 32, 1)).astype(np.float32)
    x_val = np.random.randn(5, 32, 32, 3).astype(np.float32)
    y_val = np.random.randint(0, 2, (5, 32, 32, 1)).astype(np.float32)
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(5)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(5)
    
    callback = NASMonitorCallback(validation_data=val_ds, output_dir=temp_dir, layers_to_monitor=['conv_seg'])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(train_ds, epochs=2, callbacks=[callback], verbose=0)
    
    # Should work with 4D tensors
    assert len(callback.redundancy_history) == 2
    assert os.path.exists(os.path.join(temp_dir, "metrics.csv"))


def test_csv_header_format(simple_model, dummy_dataset, temp_dir):
    """Test that CSV has correct header format."""
    train_ds, val_ds = dummy_dataset
    csv_path = os.path.join(temp_dir, "metrics.csv")
    
    callback = NASMonitorCallback(validation_data=val_ds, output_dir=temp_dir, layers_to_monitor=['conv1'])
    
    simple_model.compile(optimizer='adam', loss='binary_crossentropy')
    simple_model.fit(train_ds, epochs=1, callbacks=[callback], verbose=0)
    
    # Read CSV and check headers
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    expected_columns = ['epoch', 'conv1_redundancy_score', 'conv1_condition_number', 'conv1_rank', 'conv1_num_channels']
    
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"


def test_multiple_epochs_accumulation(simple_model, dummy_dataset, temp_dir):
    """Test that metrics accumulate correctly over multiple epochs."""
    train_ds, val_ds = dummy_dataset
    csv_path = os.path.join(temp_dir, "metrics.csv")
    
    callback = NASMonitorCallback(validation_data=val_ds, output_dir=temp_dir, layers_to_monitor=['conv1'])
    
    simple_model.compile(optimizer='adam', loss='binary_crossentropy')
    simple_model.fit(train_ds, epochs=5, callbacks=[callback], verbose=0)
    
    # Check history length
    assert len(callback.redundancy_history) == 5
    
    # Check epochs are sequential
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert list(df['epoch']) == [0, 1, 2, 3, 4]


def test_callback_reset_on_new_training(simple_model, dummy_dataset, temp_dir):
    """Test that callback state resets properly for new training run."""
    train_ds, val_ds = dummy_dataset
    
    callback = NASMonitorCallback(validation_data=val_ds, output_dir=temp_dir, layers_to_monitor=['conv1'])
    
    simple_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # First training
    simple_model.fit(train_ds, epochs=2, callbacks=[callback], verbose=0)
    first_history_len = len(callback.redundancy_history)
    
    # Second training (should continue accumulating in same callback instance)
    simple_model.fit(train_ds, epochs=3, callbacks=[callback], verbose=0)
    
    # History should have both trainings
    assert len(callback.redundancy_history) == first_history_len + 3


@pytest.mark.parametrize("monitor_frequency", ["epoch", "batch"])
def test_different_log_frequencies(simple_model, dummy_dataset, temp_dir, monitor_frequency):
    """Test callback with different logging frequencies."""
    train_ds, val_ds = dummy_dataset
    
    callback = NASMonitorCallback(
            layers_to_monitor=['conv1'],
            log_frequency=1,
            output_dir=temp_dir,
            monitor_frequency=monitor_frequency,
            validation_data=val_ds
        )
    
    simple_model.compile(optimizer='adam', loss='binary_crossentropy')
    simple_model.fit(train_ds, epochs=2, callbacks=[callback], verbose=0)
    
    assert len(callback.redundancy_history) > 0
    if monitor_frequency == 'epoch':
        assert 'epoch' in callback.redundancy_history[0]
    else:
        assert 'step' in callback.redundancy_history[0]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
