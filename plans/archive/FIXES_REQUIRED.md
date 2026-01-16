# Required Fixes for Nano-U Project

This document provides actionable fixes for the issues identified in CODE_REVIEW.md.

## CRITICAL: Missing Model Implementation Files

**Priority**: IMMEDIATE - Code cannot run without these files.

### What's Missing

The project references two model files that don't exist:
- `src/models/Nano_U/model_tf.py`
- `src/models/BU_Net/model_tf.py`

### Required Implementation

Each file needs to implement:

1. **NanoU/BUNet Model Class** - A TensorFlow/Keras model class
2. **build_nano_u()/build_bu_net() Function** - Factory function to create the model

### Example Structure (Nano_U)

```python
# src/models/Nano_U/model_tf.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class NanoU(Model):
    """Ultra-lightweight U-Net variant for embedded deployment."""
    
    def __init__(self, filters, bottleneck, decoder_filters, input_shape=(48, 64, 3)):
        super(NanoU, self).__init__()
        # TODO: Implement encoder blocks
        # TODO: Implement bottleneck
        # TODO: Implement decoder blocks with skip connections
        # TODO: Implement output layer
        
    def call(self, inputs, training=False):
        # TODO: Implement forward pass
        pass

def build_nano_u(input_shape=(48, 64, 3), filters=[8, 16, 32], 
                 bottleneck=32, decoder_filters=[16, 8]):
    """
    Build Nano-U segmentation model.
    
    Args:
        input_shape: Input image shape (H, W, C)
        filters: List of filter counts for encoder blocks
        bottleneck: Number of filters in bottleneck layer
        decoder_filters: List of filter counts for decoder blocks
        
    Returns:
        Compiled Keras Model
    """
    # TODO: Implement model construction logic
    # Should create encoder-decoder architecture with skip connections
    # Should match the architecture expected by training code
    pass
```

### Reference Architecture

Based on the config file, expected architectures:

**BU_Net (Teacher)**:
- Encoder filters: [64, 128, 256, 512, 1024, 2048]
- Bottleneck: 2048 filters
- Decoder filters: [1024, 512, 256, 128, 64]
- Input: (48, 64, 3)
- Output: (48, 64, 1) with logits (no sigmoid)

**Nano_U (Student)**:
- Encoder filters: [8, 16, 32]
- Bottleneck: 32 filters
- Decoder filters: [16, 8]
- Input: (48, 64, 3)
- Output: (48, 64, 1) with logits (no sigmoid)

### Important Notes

1. **Output Format**: Models should output **logits** (raw scores), not probabilities
   - The training code applies sigmoid: `tf.math.sigmoid(logits)`
   - Loss uses `BinaryCrossentropy(from_logits=True)`

2. **Skip Connections**: U-Net architectures require skip connections between encoder and decoder

3. **Consistent Naming**: Layer names should follow patterns for NAS analysis:
   - Use prefixes like "encoder_", "decoder_", "bottleneck_"
   - This enables covariance analysis in `nas_covariance.py`

---

## HIGH PRIORITY: Input Validation

### Location: `src/train.py`

Add validation at the start of the `train()` function:

```python
def train(model_name="nano_u", epochs=None, batch_size=None, lr=None,
          distill=False, teacher_weights=None, alpha=None, temperature=None,
          augment=True, config_path="config/config.yaml"):
    
    # Input validation
    if epochs is not None and epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size is not None and batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if lr is not None and (lr <= 0 or lr > 1):
        raise ValueError(f"learning_rate must be in (0, 1], got {lr}")
    if alpha is not None and not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if temperature is not None and temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    
    # Rest of function...
```

---

## HIGH PRIORITY: Fix Error Handling

### Location: `src/train.py` lines 99-103, 117-118

**Current (Bad)**:
```python
try:
    m.update_state(y, student_predictions)
except Exception:
    print(f"Warning: failed to update metric {m}")
```

**Fixed**:
```python
import logging

logger = logging.getLogger(__name__)

try:
    m.update_state(y, student_predictions)
except (ValueError, TypeError) as e:
    # Only catch expected exceptions from metric updates
    logger.warning(f"Failed to update metric {m.name}: {e}")
except Exception as e:
    # Log unexpected exceptions with full details
    logger.error(f"Unexpected error updating metric {m.name}: {e}", exc_info=True)
    raise  # Re-raise unexpected errors
```

---

## HIGH PRIORITY: Path Traversal Protection

### Location: `src/utils/config.py`

**Current**:
```python
def _resolve_path(p, project_root):
    if not isinstance(p, str):
        return p
    path = Path(p)
    resolved = path if path.is_absolute() else (project_root / path)
    return str(resolved)
```

**Fixed**:
```python
def _resolve_path(p, project_root):
    if not isinstance(p, str):
        return p
    path = Path(p)
    resolved = path if path.is_absolute() else (project_root / path)
    
    # Security: Ensure resolved path is within project root
    try:
        resolved = resolved.resolve()
        project_root_resolved = Path(project_root).resolve()
        if not str(resolved).startswith(str(project_root_resolved)):
            raise ValueError(f"Path '{p}' resolves outside project root")
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path '{p}': {e}")
    
    return str(resolved)
```

---

## HIGH PRIORITY: Update Deprecated API

### Location: `src/train.py` line 39

**Current**:
```python
tf.config.experimental.set_memory_growth(gpu, True)
```

**Fixed**:
```python
tf.config.set_memory_growth(gpu, True)
```

---

## MEDIUM PRIORITY: Add Type Hints

### Example: `src/utils/data.py`

**Current**:
```python
def make_dataset(img_files, mask_files, batch_size=8, shuffle=True, augment=False,
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                 flip_prob=0.5, max_rotation_deg=20,
                 brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
```

**Fixed**:
```python
from typing import List, Tuple
import tensorflow as tf

def make_dataset(
    img_files: List[str],
    mask_files: List[str],
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = False,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
    flip_prob: float = 0.5,
    max_rotation_deg: int = 20,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.05
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset for image segmentation.
    
    Args:
        img_files: List of paths to input images
        mask_files: List of paths to corresponding masks
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation
        mean: RGB mean values for normalization
        std: RGB standard deviation values for normalization
        flip_prob: Probability of horizontal flip (0-1)
        max_rotation_deg: Maximum rotation angle in degrees
        brightness: Brightness adjustment range
        contrast: Contrast adjustment range
        saturation: Saturation adjustment range
        hue: Hue adjustment range
        
    Returns:
        TensorFlow dataset yielding (image, mask) batches
        
    Raises:
        ValueError: If file lists are empty or mismatched
    """
```

---

## MEDIUM PRIORITY: Resource Cleanup

### Location: `src/nas_covariance.py`

**Add Context Manager Support**:

```python
class ActivationExtractor:
    """Cached extractor for intermediate activations."""
    
    def __init__(self, model: tf.keras.Model, layer_selectors: List[Union[str, tf.keras.layers.Layer]]):
        self.model = model
        self.layer_selectors = layer_selectors
        self.layer_names = self._resolve_layer_names()
        self.intermediate_model = self._build_intermediate_model()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def cleanup(self):
        """Release cached intermediate model to free memory."""
        if hasattr(self, 'intermediate_model'):
            del self.intermediate_model
```

**Usage**:
```python
with ActivationExtractor(model, layer_names) as extractor:
    activations = extractor(inputs)
    # Use activations...
# Automatically cleaned up
```

---

## MEDIUM PRIORITY: Configuration Validation

### Add Schema Validation

Install: `pip install pydantic`

**Create** `src/utils/config_schema.py`:
```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict

class DataPathsConfig(BaseModel):
    models_dir: str
    
class DataConfig(BaseModel):
    input_shape: List[int] = Field(min_length=3, max_length=3)
    num_classes: int = Field(ge=1)
    paths: DataPathsConfig
    
    @field_validator('input_shape')
    def validate_shape(cls, v):
        if not all(x > 0 for x in v):
            raise ValueError("All dimensions must be positive")
        return v

class Config(BaseModel):
    data: DataConfig
    # Add other sections...

def validate_config(config_dict: dict) -> Config:
    """Validate configuration dictionary against schema."""
    return Config(**config_dict)
```

**Update** `src/utils/config.py`:
```python
from .config_schema import validate_config

def load_config(config_path="config/config.yaml"):
    # ... existing loading code ...
    
    # Validate configuration
    try:
        validated_config = validate_config(config)
        return validated_config.model_dump()
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")
```

---

## MEDIUM PRIORITY: Improve Data Loading Error Handling

### Location: `src/utils/data.py`

**Better Error Handling in Dataset Pipeline**:

```python
def make_dataset(img_files, mask_files, batch_size=8, shuffle=True, augment=False,
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                 flip_prob=0.5, max_rotation_deg=20,
                 brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
    
    # Validate inputs early
    if len(img_files) == 0:
        raise ValueError("make_dataset: img_files list is empty")
    if len(mask_files) == 0:
        raise ValueError("make_dataset: mask_files list is empty")
    
    # Validate files exist before creating dataset
    missing_imgs = [f for f in img_files if not os.path.exists(f)]
    missing_masks = [f for f in mask_files if not os.path.exists(f)]
    
    if missing_imgs:
        raise FileNotFoundError(
            f"Missing {len(missing_imgs)} image files. "
            f"First missing: {missing_imgs[0]}"
        )
    if missing_masks:
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} mask files. "
            f"First missing: {missing_masks[0]}"
        )
    
    # Rest of function...
```

---

## LOW PRIORITY: Extract Magic Numbers

### Create Constants File

**Create** `src/constants.py`:
```python
"""Project-wide constants."""

# Segmentation
BINARY_THRESHOLD = 0.5

# Epsilon for numerical stability
EPSILON = 1e-7

# Default augmentation parameters
DEFAULT_FLIP_PROB = 0.5
DEFAULT_MAX_ROTATION = 20
DEFAULT_BRIGHTNESS = 0.2
DEFAULT_CONTRAST = 0.2
DEFAULT_SATURATION = 0.2
DEFAULT_HUE = 0.05

# Default normalization (maps [0, 255] to [-1, 1])
DEFAULT_MEAN = [0.5, 0.5, 0.5]
DEFAULT_STD = [0.5, 0.5, 0.5]
```

**Usage Example**:
```python
from src.constants import BINARY_THRESHOLD, EPSILON

# In src/evaluate.py
mask = tf.cast(mask > BINARY_THRESHOLD, tf.float32)
```

---

## LOW PRIORITY: Break Down Large Functions

### Example: `src/train.py`

The `train()` function should be split into smaller functions:

```python
def _setup_data_paths(config):
    """Extract and resolve data paths from config."""
    # Lines 231-238 of current train()
    
def _load_datasets(train_files, val_files, config, batch_size, augment):
    """Create training and validation datasets."""
    # Lines 240-255 of current train()
    
def _setup_optimizer(train_cfg):
    """Create and configure optimizer."""
    # Lines 261-270 of current train()
    
def _setup_distillation(student, teacher_weights, config):
    """Setup distillation training."""
    # Lines 286-308 of current train()
    
def train(model_name="nano_u", ...):
    """Main training entrypoint."""
    config = load_config(config_path)
    
    # Use helper functions
    paths = _setup_data_paths(config)
    train_ds, val_ds = _load_datasets(...)
    optimizer = _setup_optimizer(...)
    
    # Shorter, clearer main function
```

---

## Summary of Implementation Priority

### Week 1: Critical Issues
1. ✅ Create `model_tf.py` files for both models
2. ✅ Add input validation to `train()`, `quantize()`, and other entry points
3. ✅ Fix error handling (logging, specific exceptions)

### Week 2: Security & Quality
4. ✅ Add path traversal protection
5. ✅ Update deprecated TensorFlow APIs
6. ✅ Improve data loading error handling
7. ✅ Add type hints to public APIs

### Week 3: Testing & Documentation
8. ✅ Add test cases for critical paths
9. ✅ Add configuration schema validation
10. ✅ Improve function documentation

### Week 4: Refactoring
11. ✅ Extract constants
12. ✅ Break down large functions
13. ✅ Add context managers for resource cleanup

---

## Testing Checklist

After implementing fixes, verify:

- [ ] `python src/train.py --model nano_u --epochs 1` runs without errors
- [ ] `python src/prepare_data.py` successfully processes data
- [ ] `python src/quantize.py --model-name nano_u` creates TFLite file
- [ ] `python src/evaluate.py --model-name nano_u` generates predictions
- [ ] `pytest tests/` passes all tests
- [ ] Type checking with `mypy src/` shows no errors (after adding type hints)

---

## Questions for Project Owner

1. What is the expected architecture for NanoU and BUNet models? 
   - Should they follow standard U-Net with skip connections?
   - Any specific requirements for layer naming or structure?

2. Are there any pre-trained weights available for testing?

3. Is there a preferred testing framework (pytest, unittest)?

4. Should I add pre-commit hooks for linting/formatting?

5. Are there any deployment constraints for the ESP32 that affect model design?
