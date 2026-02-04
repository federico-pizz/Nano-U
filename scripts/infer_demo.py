#!/usr/bin/env python3
"""
Inference Demo Script for Nano-U Model

This script demonstrates how to:
1. Load a trained Nano-U model
2. Prepare input images for inference
3. Generate segmentation masks
4. Visualize results

Usage:
    python scripts/infer_demo.py --model models/nano_u.keras --image path/to/image.png
    python scripts/infer_demo.py --model models/nano_u_int8.tflite --image path/to/image.png --quantized
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def normalize_image(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Normalize image to [-1, 1] range.
    
    Args:
        image: Input image (H, W, 3) with values [0, 255]
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Normalized image (H, W, 3) with values [-1, 1]
    """
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    return image


def denormalize_mask(mask):
    """
    Convert normalized mask [any range] to [0, 255] for visualization.
    
    Args:
        mask: Output from model (logits or probabilities)
    
    Returns:
        uint8 mask [0, 255]
    """
    if mask.max() > 1.0:
        # Logits: apply sigmoid
        mask = 1.0 / (1.0 + np.exp(-mask))
    
    # Convert to [0, 255]
    mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
    return mask


def infer_keras(model_path, image, threshold=0.5):
    """
    Run inference with Keras model.
    
    Args:
        model_path: Path to .keras model file
        image: Input image (H, W, 3) [0, 255]
        threshold: Binary segmentation threshold [0, 1]
    
    Returns:
        Tuple of (logits, probabilities, binary_mask)
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow not installed. Install with: pip install tensorflow")
        sys.exit(1)
    
    # Load model
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Prepare input
    image_normalized = normalize_image(image)
    input_batch = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
    
    # Inference
    print("Running inference...")
    logits = model(input_batch, training=False).numpy()[0]  # (H, W, 1)
    
    # Convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
    
    # Binary mask with threshold
    binary_mask = (probs > threshold).astype(np.uint8) * 255
    
    return logits, probs, binary_mask


def infer_tflite(model_path, image, threshold=0.5):
    """
    Run inference with TFLite model (quantized).
    
    Args:
        model_path: Path to .tflite model file
        image: Input image (H, W, 3) [0, 255]
        threshold: Binary segmentation threshold [0, 1]
    
    Returns:
        Tuple of (logits, probabilities, binary_mask)
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow not installed. Install with: pip install tensorflow")
        sys.exit(1)
    
    # Load interpreter
    print(f"Loading TFLite model from {model_path}...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Prepare input
    image_normalized = normalize_image(image)
    input_data = np.expand_dims(image_normalized, axis=0).astype(input_details['dtype'])
    
    # Inference
    print("Running inference (quantized)...")
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details['index'])[0]
    
    # For INT8, scale back to float
    if logits.dtype == np.int8:
        # Dequantize: INT8 → float
        scale = output_details['quantization'][0]
        zero_point = output_details['quantization'][1]
        logits = (logits.astype(np.float32) - zero_point) * scale
    
    # Convert to probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    # Binary mask
    binary_mask = (probs > threshold).astype(np.uint8) * 255
    
    return logits, probs, binary_mask


def visualize_result(original_image, mask, output_path=None, colormap='jet'):
    """
    Create visualization with original image and segmentation mask.
    
    Args:
        original_image: Original image (H, W, 3) [0, 255]
        mask: Segmentation mask (H, W) [0, 255]
        output_path: Optional path to save visualization
        colormap: OpenCV colormap name (jet, viridis, turbo, etc.)
    
    Returns:
        Visualization image (H, W*2, 3)
    """
    # Convert mask to 3-channel for visualization
    mask_colorized = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    # Create overlay: 50% original + 50% mask
    overlay = cv2.addWeighted(original_image, 0.5, mask_colorized, 0.5, 0)
    
    # Concatenate original and overlay side-by-side
    result = np.concatenate([original_image, overlay], axis=1)
    
    # Save if requested
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Visualization saved to {output_path}")
    
    return result


def main(args):
    """Main inference demo."""
    
    # Check inputs
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    # Load image
    print(f"Loading image from {args.image}...")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        sys.exit(1)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB
    original_shape = image.shape
    print(f"Image shape: {original_shape}")
    
    # Resize to model input size (48x64)
    image_resized = cv2.resize(image, (64, 48))  # (W, H) for OpenCV
    print(f"Resized to: {image_resized.shape}")
    
    # Run inference
    if args.quantized:
        logits, probs, binary_mask = infer_tflite(args.model, image_resized, args.threshold)
    else:
        logits, probs, binary_mask = infer_keras(args.model, image_resized, args.threshold)
    
    # Print statistics
    print(f"\nInference Results:")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"  Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  Mean probability: {probs.mean():.3f}")
    print(f"  Mask coverage: {(binary_mask > 0).sum() / binary_mask.size * 100:.1f}%")
    
    # Visualize
    if args.output:
        visualize_result(image_resized, binary_mask.squeeze(), output_path=args.output)
        print(f"\nSegmentation mask saved to {args.output}")
    
    # Save raw mask if requested
    if args.mask_output:
        cv2.imwrite(args.mask_output, binary_mask.squeeze())
        print(f"Raw mask saved to {args.mask_output}")
    
    # Interactive visualization (optional)
    if args.display:
        print("\nDisplaying results. Press any key to close.")
        vis = visualize_result(image_resized, binary_mask.squeeze())
        cv2.imshow('Segmentation', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference demo for Nano-U segmentation model"
    )
    parser.add_argument("--model", required=True, help="Path to model (.keras or .tflite)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--quantized", action="store_true", help="Use TFLite quantized model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary mask threshold [0, 1]")
    parser.add_argument("--output", help="Path to save visualization")
    parser.add_argument("--mask-output", help="Path to save raw binary mask")
    parser.add_argument("--display", action="store_true", help="Display result interactively")
    
    args = parser.parse_args()
    main(args)
