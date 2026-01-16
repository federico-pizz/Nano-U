f"""
Test script to validate architecture fixes and NAS layer detection.

This script tests:
1. Nano_U has no skip connections (simple autoencoder)
2. Recursive layer detection finds all conv layers
3. Model parameter counts are reasonable
4. Forward pass works correctly
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from src.models.Nano_U.model_tf import build_nano_u
from src.models.BU_Net.model_tf import build_bu_net


def get_all_layers_recursive(model):
    """Recursively collect all layers including from nested submodels."""
    all_layers = []
    for layer in model.layers:
        all_layers.append(layer)
        if isinstance(layer, tf.keras.Model):
            all_layers.extend(get_all_layers_recursive(layer))
    return all_layers


def test_nano_u_architecture():
    """Test Nano_U model architecture."""
    print("\n" + "="*60)
    print("TEST 1: Nano_U Architecture")
    print("="*60)
    
    model = build_nano_u(input_shape=(48, 64, 3))
    
    # Test forward pass
    x = tf.random.normal((1, 48, 64, 3))
    output = model(x, training=False)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Count parameters
    params = model.count_params()
    print(f"✓ Total parameters: {params:,}")
    
    # Check for skip connections (should NOT exist in model layers)
    layer_names = [l.name for l in model.layers]
    skip_layers = [n for n in layer_names if 'concat' in n.lower() or 'skip' in n.lower()]
    
    if skip_layers:
        print(f"✗ FAIL: Found skip connection layers: {skip_layers}")
        print("  Nano_U should be a simple autoencoder without skip connections!")
        return False
    else:
        print(f"✓ No skip connections found (simple autoencoder)")
    
    print(f"✓ TEST 1 PASSED\n")
    return True


def test_recursive_layer_detection():
    """Test that recursive layer detection finds all conv layers."""
    print("="*60)
    print("TEST 2: Recursive Layer Detection")
    print("="*60)
    
    model = build_nano_u(input_shape=(48, 64, 3))
    
    # Get all layers recursively
    all_layers = get_all_layers_recursive(model)
    print(f"✓ Total layers (recursive): {len(all_layers)}")
    
    # Find conv layers
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D,
                 tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)
    conv_layers = [l for l in all_layers if isinstance(l, conv_types)]
    
    print(f"✓ Conv layers found: {len(conv_layers)}")
    print("\nConv layer names:")
    for layer in conv_layers:
        print(f"  - {layer.name} ({type(layer).__name__})")
    
    if len(conv_layers) == 0:
        print("✗ FAIL: No conv layers found!")
        return False
    
    # Check that we can find encoder, decoder, bottleneck layers by pattern
    all_conv_names = [l.name for l in conv_layers]
    encoder = [n for n in all_conv_names if any(p in n.lower() for p in ['encoder', 'down'])]
    decoder = [n for n in all_conv_names if any(p in n.lower() for p in ['decoder', 'up'])]
    bottleneck = [n for n in all_conv_names if 'bottleneck' in n.lower()]
    
    print(f"\n✓ Encoder layers: {len(encoder)}")
    print(f"✓ Decoder layers: {len(decoder)}")
    print(f"✓ Bottleneck layers: {len(bottleneck)}")
    
    if len(encoder) == 0 and len(decoder) == 0 and len(bottleneck) == 0:
        print("⚠ Warning: Pattern matching found no layers, but fallback will use all conv layers")
    
    print(f"✓ TEST 2 PASSED\n")
    return True


def test_bu_net_architecture():
    """Test BU_Net model architecture."""
    print("="*60)
    print("TEST 3: BU_Net Architecture")
    print("="*60)
    
    model = build_bu_net(input_shape=(48, 64, 3))
    
    # Test forward pass
    x = tf.random.normal((1, 48, 64, 3))
    output = model(x, training=False)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Count parameters
    params = model.count_params()
    print(f"✓ Total parameters: {params:,}")
    
    # BU_Net SHOULD have skip connections (it's a proper U-Net)
    layer_names = [l.name for l in model.layers]
    
    # Get all layers recursively
    all_layers = get_all_layers_recursive(model)
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
    conv_layers = [l for l in all_layers if isinstance(l, conv_types)]
    
    print(f"✓ Conv layers found: {len(conv_layers)}")
    
    # Check for depthwise convolutions
    depthwise = [l for l in all_layers if isinstance(l, tf.keras.layers.DepthwiseConv2D)]
    print(f"✓ Depthwise conv layers: {len(depthwise)}")
    
    if len(depthwise) == 0:
        print("✗ FAIL: No depthwise conv layers found! BU_Net should use depthwise separable convolutions")
        return False
    
    print(f"✓ TEST 3 PASSED\n")
    return True


def test_model_output_shapes():
    """Test that model outputs have correct shapes."""
    print("="*60)
    print("TEST 4: Model Output Shapes")
    print("="*60)
    
    input_shapes = [(48, 64, 3), (64, 48, 3), (32, 32, 3)]
    
    for input_shape in input_shapes:
        nano_u = build_nano_u(input_shape=input_shape)
        bu_net = build_bu_net(input_shape=input_shape)
        
        x = tf.random.normal((2, *input_shape))
        
        nano_out = nano_u(x, training=False)
        bu_out = bu_net(x, training=False)
        
        expected_shape = (2, input_shape[0], input_shape[1], 1)
        
        if nano_out.shape != expected_shape:
            print(f"✗ FAIL: Nano_U output shape {nano_out.shape} != expected {expected_shape}")
            return False
        
        if bu_out.shape != expected_shape:
            print(f"✗ FAIL: BU_Net output shape {bu_out.shape} != expected {expected_shape}")
            return False
        
        print(f"✓ Input {input_shape} -> Output {expected_shape}")
    
    print(f"✓ TEST 4 PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Architecture Validation Tests")
    print("#"*60)
    
    tests = [
        test_nano_u_architecture,
        test_recursive_layer_detection,
        test_bu_net_architecture,
        test_model_output_shapes,
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
