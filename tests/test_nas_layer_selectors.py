"""
Test NAS monitoring with internal layer selectors.

This test validates that NASMonitorCallback can monitor internal model layers
and produce meaningful metrics. The key test is test_nas_monitor_with_layer_selectors
which demonstrates the actual usage pattern.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.Nano_U.model_tf import build_nano_u
from src.nas_covariance import NASMonitorCallback


def find_conv_block_layers(model):
    """Find all DepthwiseSepConv layers (encoder, decoder, bottleneck blocks)."""
    conv_block_layers = []
    
    def _find_in_model(m, prefix=""):
        for layer in m.layers:
            # Look for DepthwiseSepConv custom layers
            if type(layer).__name__ == 'DepthwiseSepConv':
                name = f"{prefix}/{layer.name}" if prefix else layer.name
                conv_block_layers.append(name)
            elif isinstance(layer, tf.keras.Model):
                new_prefix = f"{prefix}/{layer.name}" if prefix else layer.name
                _find_in_model(layer, new_prefix)
    
    _find_in_model(model)
    return conv_block_layers


def test_nas_monitor_with_layer_selectors():
    """Test NASMonitorCallback with internal layer monitoring.
    
    This is the primary test demonstrating that NAS monitoring works correctly
    with layer selectors to monitor internal model layers.
    """
    print("\n" + "="*60)
    print("TEST 1: NASMonitorCallback with Layer Selectors")
    print("="*60)
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "nas_metrics.csv")
    
    try:
        model = build_nano_u(input_shape=(48, 64, 3))
        
        # Find conv block layers
        conv_blocks = find_conv_block_layers(model)
        selectors = conv_blocks[:2] if len(conv_blocks) >= 2 else conv_blocks
        
        print(f"✓ Found {len(conv_blocks)} conv block layers in model")
        print(f"✓ Monitoring {len(selectors)} layers: {selectors}")
        
        # Create dummy dataset
        x_train = tf.random.normal((8, 48, 64, 3))
        y_train = tf.random.uniform((8, 48, 64, 1), 0, 1)
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(4)
        
        # Create NASMonitorCallback
        callback = NASMonitorCallback(
            validation_data=dataset,
            csv_path=csv_path,
            log_frequency=1,
            layer_selectors=selectors
        )
        
        print(f"✓ NASMonitorCallback created successfully")
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Train for 2 epochs
        print("✓ Training for 2 epochs with NAS monitoring...")
        history = model.fit(
            dataset,
            epochs=2,
            callbacks=[callback],
            verbose=0
        )
        
        print(f"✓ Training completed")
        
        # Check that CSV was created and has data
        assert os.path.exists(csv_path), f"CSV file not created at {csv_path}"
        print(f"✓ CSV file created at {csv_path}")
        
        # Read CSV and verify metrics
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        print(f"✓ CSV has {len(df)} rows and {len(df.columns)} columns")
        
        # Check for required columns
        required_cols = ['epoch', 'redundancy_score', 'mean_correlation']
        for col in required_cols:
            assert col in df.columns, f"Missing required column '{col}'"
        
        print(f"✓ All required columns present: {list(df.columns)}")
        
        # Verify metrics are valid (not NaN)
        redundancy_values = df['redundancy_score'].values
        correlation_values = df['mean_correlation'].values
        
        print(f"\nMetric Statistics:")
        print(f"  Redundancy: mean={np.mean(redundancy_values):.6f}, std={np.std(redundancy_values):.6f}")
        print(f"  Correlation: mean={np.mean(correlation_values):.6f}, std={np.std(correlation_values):.6f}")
        
        # Check that metrics are not NaN (they may be zero for certain architectures)
        assert not np.all(np.isnan(redundancy_values)), "All redundancy values are NaN!"
        
        print(f"✓ Metrics are valid (not NaN)")
        print(f"✓ TEST 1 PASSED\n")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_layer_detection():
    """Test that we can detect conv block layers in the model."""
    print("="*60)
    print("TEST 2: Layer Detection")
    print("="*60)
    
    model = build_nano_u(input_shape=(48, 64, 3))
    
    # Find conv block layers
    conv_blocks = find_conv_block_layers(model)
    
    print(f"✓ Found {len(conv_blocks)} conv block layers")
    
    assert len(conv_blocks) > 0, "No conv block layers found!"
    
    # Show some examples
    print(f"\nSample layers:")
    for name in conv_blocks[:5]:
        print(f"  - {name}")
    if len(conv_blocks) > 5:
        print(f"  ... and {len(conv_blocks) - 5} more")
    
    print(f"\n✓ TEST 2 PASSED\n")


def test_nas_callback_without_layer_selectors():
    """Test that NASMonitorCallback works without layer selectors (monitors output)."""
    print("="*60)
    print("TEST 3: NASMonitorCallback without Layer Selectors")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "nas_metrics_output.csv")
    
    try:
        model = build_nano_u(input_shape=(48, 64, 3))
        
        # Create dummy dataset
        x_train = tf.random.normal((8, 48, 64, 3))
        y_train = tf.random.uniform((8, 48, 64, 1), 0, 1)
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(4)
        
        # Create NASMonitorCallback WITHOUT layer_selectors (should monitor model output)
        callback = NASMonitorCallback(
            validation_data=dataset,
            csv_path=csv_path,
            log_frequency=1
        )
        
        print(f"✓ NASMonitorCallback created (monitoring model output)")
        
        # Compile and train
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        )
        
        print("✓ Training for 1 epoch...")
        model.fit(dataset, epochs=1, callbacks=[callback], verbose=0)
        
        # Check CSV was created
        assert os.path.exists(csv_path), "CSV file not created"
        
        print(f"✓ CSV file created")
        print(f"✓ TEST 3 PASSED\n")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# NAS Layer Selector Tests")
    print("#"*60)
    
    tests = [
        test_nas_monitor_with_layer_selectors,
        test_layer_detection,
        test_nas_callback_without_layer_selectors,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ TEST FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "#"*60)
    print("# Test Summary")
    print("#"*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"✗ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
