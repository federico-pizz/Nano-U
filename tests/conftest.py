"""Shared fixtures for the Nano-U test suite."""

import os
import tempfile

import cv2
import numpy as np
import pytest
import tensorflow as tf

# Enable GPU memory growth before any op initializes the device. The suite mixes
# GPU forward/training passes with INT8 TFLite conversion; without growth, TF
# pre-allocates the whole GPU and the converter can intermittently fail under the
# resulting memory pressure. Harmless on CPU-only machines.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # device already initialized; nothing we can do this late

H, W = 60, 80


@pytest.fixture(scope="session")
def img_tensor():
    """Normalised float32 image batch (1, H, W, 3) in ~[-1, 1]."""
    rng = np.random.default_rng(0)
    return tf.constant(rng.standard_normal((1, H, W, 3)).astype(np.float32))


@pytest.fixture(scope="session")
def mask_tensor():
    """Binary float32 mask batch (1, H, W, 1)."""
    rng = np.random.default_rng(1)
    return tf.constant((rng.random((1, H, W, 1)) > 0.5).astype(np.float32))


@pytest.fixture(scope="session")
def logits_tensor():
    """Raw logit batch (1, H, W, 1) — values outside [0, 1]."""
    rng = np.random.default_rng(2)
    return tf.constant(rng.standard_normal((1, H, W, 1)).astype(np.float32) * 3.0)


@pytest.fixture(scope="session")
def tmp_png_dataset(tmp_path_factory):
    """Creates a tiny on-disk dataset: 4 images + 4 masks as PNG files."""
    base = tmp_path_factory.mktemp("dataset")
    img_dir = base / "img"
    mask_dir = base / "mask"
    img_dir.mkdir()
    mask_dir.mkdir()

    rng = np.random.default_rng(42)
    paths = {"imgs": [], "masks": []}
    for i in range(4):
        img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
        mask = ((rng.random((H, W)) > 0.5) * 255).astype(np.uint8)
        img_path = str(img_dir / f"frame_{i:03d}.png")
        mask_path = str(mask_dir / f"frame_{i:03d}.png")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)
        paths["imgs"].append(img_path)
        paths["masks"].append(mask_path)

    return paths


@pytest.fixture(scope="session")
def nano_u_model():
    """A small Nano-U instance (16-channel) built once per test session."""
    from src.models.builders import create_nano_u
    return create_nano_u(input_shape=(H, W, 3), filters=[4, 8, 16], bottleneck=16)
