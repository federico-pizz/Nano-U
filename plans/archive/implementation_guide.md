# Implementation Guide - Nano-U Simplification & Bug Fixes

## Overview
This guide provides specific code changes needed to fix the identified issues.

---

## Issue #1: Nano_U Skip Connections (CRITICAL)

### Problem
Current implementation has U-Net style skip connections, but old PyTorch implementation was a simple autoencoder WITHOUT skip connections.

### Location
`src/models/Nano_U/model_tf.py` - Lines 84-110

### Current Code (WITH skip connections)
```python
def call(self, inputs, training=False):
    x = inputs
    skips = []
    
    # Encoder
    for i, conv in enumerate(self.enc_convs):
        x = conv(x, training=training)
        if i < self.num_levels - 1:
            skips.append(x)  # ← SAVES SKIP CONNECTIONS
            x = self.enc_pools[i](x)

    # Bottleneck
    x = self.bottleneck(x, training=training)

    # Decoder with skip connections
    for up, conv in zip(self.dec_ups, self.dec_convs):
        x = up(x)
        if skips:
            skip = skips.pop()  # ← USES SKIP CONNECTIONS
            if tf.shape(x)[1] != tf.shape(skip)[1] or tf.shape(x)[2] != tf.shape(skip)[2]:
                skip = tf.image.resize(skip, size=(tf.shape(x)[1], tf.shape(x)[2]), method='bilinear')
            x = layers.Concatenate(name=f'skip_concat_{len(skips)}')([x, skip])
        x = conv(x, training=training)

    logits = self.output_conv(x)
    return logits
```

### Fixed Code (WITHOUT skip connections - matches old)
```python
def call(self, inputs, training=False):
    x = inputs
    
    # Encoder - NO skip saving
    for i, conv in enumerate(self.enc_convs):
        x = conv(x, training=training)
        if i < self.num_levels - 1:
            x = self.enc_pools[i](x)

    # Bottleneck
    x = self.bottleneck(x, training=training)

    # Decoder - NO skip connections
    for up, conv in zip(self.dec_ups, self.dec_convs):
        x = up(x)
        x = conv(x, training=training)

    logits = self.output_conv(x)
    return logits
```

### Additional Changes Needed
Since we're removing skip connections, we need to ensure the decoder input channels are correct:

In `__init__` method around line 75-79:
```python
# OLD (expects concatenated skip + upsampled):
self.dec_convs.append(DepthwiseSepConv(f, stride=1, name=f'decoder_conv_{i}'))

# NEW (only upsampled input, no concatenation):
# First decoder layer takes bottleneck channels as input
# Subsequent layers take previous decoder output
# This is already correct in the current DepthwiseSepConv since it adapts to input
```

Actually, looking at DepthwiseSepConv, it uses `DepthwiseConv2D` which operates on the input channels automatically, so no change needed there. Just remove skip connections!

---

## Issue #2: train_with_nas.py Layer Detection Bug (CRITICAL)

### Problem Location
`src/train_with_nas.py` - Lines 348-390

### Current Problematic Code
```python
# Line 356-361: Detects inner model but doesn't use it correctly
inner_model = None
for layer in student_model.layers:
    if isinstance(layer, tf.keras.Model) and layer.name in ['BU_Net', 'Nano_U']:
        inner_model = layer
        print(f"Detected nested subclassed model: {layer.name}")
        break

# Line 365: Uses wrong model for layer detection
target_model = inner_model if inner_model is not None else student_model

# Line 369-380: Auto-detect fails
all_conv = UNetLayerSelector.get_all_conv_layers(target_model)
if not all_conv:
    print(f"Warning: No conv layers found in model. Available layers: {[l.name for l in target_model.layers[:10]]}")
    selectors = []
```

### Root Cause
When `build_nano_u()` returns a functional Model wrapping NanoU:
1. The functional Model's layers include the NanoU instance
2. But NanoU's internal layers (encoder_conv_0, decoder_conv_1, etc.) are NOT in functional Model's layers list
3. They ARE in NanoU.layers, but we're searching target_model.layers
4. Result: No conv layers found → NAS disabled silently

### Fix: Recursive Layer Search

Replace lines 348-390 with:
```python
extractor = None
if enable_nas:
    # Use UNetLayerSelector to automatically find conv layers
    if nas_layers:
        selectors = [s.strip() for s in nas_layers.split(',')]
    else:
        # Recursive function to get all layers from nested models
        def get_all_layers_recursive(model):
            """Recursively collect all layers including from nested submodels."""
            all_layers = []
            for layer in model.layers:
                all_layers.append(layer)
                # If this layer is itself a Model, recursively get its layers
                if isinstance(layer, tf.keras.Model):
                    all_layers.extend(get_all_layers_recursive(layer))
            return all_layers
        
        # Get all layers recursively
        all_layers = get_all_layers_recursive(student_model)
        
        # Find conv layers
        conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D,
                     tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)
        all_conv = [l.name for l in all_layers if isinstance(l, conv_types)]
        
        if not all_conv:
            print(f"Warning: No conv layers found. Available layer types: {[type(l).__name__ for l in all_layers[:20]]}")
            selectors = []
        else:
            # Pattern-based selection for encoder, decoder, bottleneck
            encoder = [n for n in all_conv if any(p in n.lower() for p in ['encoder', 'down'])]
            decoder = [n for n in all_conv if any(p in n.lower() for p in ['decoder', 'up'])]
            bottleneck = [n for n in all_conv if 'bottleneck' in n.lower()]
            selectors = encoder + decoder + bottleneck
            
            if not selectors:
                # Fallback to all conv layers
                selectors = all_conv
                print(f"Pattern matching failed, using all conv layers: {len(selectors)} layers")
            else:
                print(f"Auto-detected NAS layers ({len(selectors)}): {selectors[:5]}..." if len(selectors) > 5 else f"Auto-detected NAS layers: {selectors}")
    
    try:
        # Use the original student_model for ActivationExtractor
        # It will find layers recursively now that we have full layer names
        extractor = ActivationExtractor(student_model, selectors)
        print(f"✓ ActivationExtractor initialized with {len(extractor.layer_names)} layers")
    except ValueError as e:
        print(f"⚠ Warning: Could not initialize ActivationExtractor: {e}")
        print("NAS regularization will be disabled for this training run.")
        extractor = None
```

### Why This Works
1. Recursively searches through all nested models
2. Collects ALL layers including those in subclassed models
3. Pattern matching works on full layer names like "BU_Net/encoder_conv_0_dw"
4. ActivationExtractor gets proper layer references from the top-level model

---

## Issue #3: ActivationExtractor Layer Name Handling

### Problem Location
`src/nas_covariance.py` - Lines 48-90 in `_resolve_layer_names()`

### Current Issue
When model has nested structure, layer names might include model prefixes like "Nano_U/encoder_conv_0_dw"

### Potential Fix (IF NEEDED)
The current implementation should actually handle this correctly since it uses `model.get_layer(name)` which accepts nested names. 

**Test first** - if ActivationExtractor still fails after the train_with_nas.py fix, then we need to update the layer name resolution.

---

## Issue #4: Missing Inner Model Parameter in NASWrapper

### Problem Location
`src/train_with_nas.py` - Lines 399 and 480

### Current Code
```python
# Line 399: DistillerWithNAS
model = DistillerWithNAS(student=student_model, teacher=teacher, extractor=extractor, 
                        nas_weight=nas_weight, inner_model=inner_model)

# Line 480: NASWrapper  
model = NASWrapper(base_model=student_model, extractor=extractor, nas_weight=nas_weight, 
                   loss_fn=bce_loss, metrics=[iou_metric], inner_model=inner_model)
```

### Issue
`inner_model` is from the old detection code and will be None after our fix. These parameters are trying to work around the layer detection issue.

### Fix
After implementing recursive layer search, `inner_model` is unnecessary:

```python
# Line 399: Remove inner_model parameter
model = DistillerWithNAS(student=student_model, teacher=teacher, extractor=extractor, 
                        nas_weight=nas_weight)

# Line 480: Remove inner_model parameter
model = NASWrapper(base_model=student_model, extractor=extractor, nas_weight=nas_weight, 
                   loss_fn=bce_loss, metrics=[iou_metric])
```

Also update the class definitions (lines 56 and 118) to remove `inner_model` parameter and related logic:

```python
# Line 56: NASWrapper.__init__
def __init__(self, base_model, extractor: ActivationExtractor, nas_weight: float, loss_fn, metrics=None):
    super(NASWrapper, self).__init__()
    self.base_model = base_model
    # Remove: self.inner_model = inner_model
    self.extractor = extractor
    # ... rest of init

# Line 118: DistillerWithNAS.__init__
def __init__(self, student, teacher, extractor: ActivationExtractor, nas_weight: float):
    super(DistillerWithNAS, self).__init__()
    self.student = student
    self.teacher = teacher
    # Remove: self.inner_model = inner_model
    self.extractor = extractor
    # ... rest of init
```

And in train_step methods, remove the inner_model calls:

```python
# OLD (lines 84-86 in NASWrapper.train_step):
if self.inner_model is not None:
    _ = self.inner_model(x, training=True)
acts = self.extractor(x, training=True)

# NEW:
acts = self.extractor(x, training=True)

# Same for DistillerWithNAS (lines 168-170)
```

---

## Issue #5: Pooling Type Mismatch (Minor)

### Problem
Old Nano_U used AvgPool2d, new uses MaxPool2D

### Location
`src/models/Nano_U/model_tf.py` - Line 69

### Current
```python
self.enc_pools.append(layers.MaxPool2D(2, name=f'encoder_pool_{i}'))
```

### Fix (to match old exactly)
```python
self.enc_pools.append(layers.AveragePooling2D(2, name=f'encoder_pool_{i}'))
```

### Assessment
This is minor and may not affect performance significantly. MaxPool tends to work better for segmentation, but if we want to match the old implementation exactly, change it.

---

## Testing Checklist

After implementing fixes:

### Test 1: Model Architecture
```bash
python -c "
from src.models.Nano_U.model_tf import build_nano_u
from src.models.BU_Net.model_tf import build_bu_net
import tensorflow as tf

nano = build_nano_u(input_shape=(48, 64, 3))
bu = build_bu_net(input_shape=(48, 64, 3))

print('Nano_U parameters:', nano.count_params())
print('BU_Net parameters:', bu.count_params())

# Test forward pass
x = tf.random.normal((1, 48, 64, 3))
nano_out = nano(x)
bu_out = bu(x)
print('Nano_U output shape:', nano_out.shape)
print('BU_Net output shape:', bu_out.shape)
"
```

Expected: No errors, output shapes should be (1, 48, 64, 1)

### Test 2: Basic Training
```bash
python src/train.py --model nano_u --epochs 1 --batch-size 2
```

Expected: Training runs without errors

### Test 3: NAS Training
```bash
python src/train_with_nas.py --model nano_u --epochs 1 --batch-size 2 --enable-nas
```

Expected: 
- "Auto-detected NAS layers" message appears
- "✓ ActivationExtractor initialized with X layers" appears
- Training completes without errors

### Test 4: Layer Detection Verification
```bash
python -c "
from src.models.Nano_U.model_tf import build_nano_u
import tensorflow as tf

model = build_nano_u()

# Recursive layer collection
def get_all_layers_recursive(m):
    layers = []
    for layer in m.layers:
        layers.append(layer)
        if isinstance(layer, tf.keras.Model):
            layers.extend(get_all_layers_recursive(layer))
    return layers

all_layers = get_all_layers_recursive(model)
conv_layers = [l for l in all_layers if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))]

print(f'Total layers: {len(all_layers)}')
print(f'Conv layers: {len(conv_layers)}')
print('Conv layer names:')
for l in conv_layers:
    print(f'  - {l.name}')
"
```

Expected: Should find multiple conv layers with names like encoder_conv_0_dw, decoder_conv_1_pw, etc.

---

## Summary of Changes

| File | Lines | Change | Priority |
|------|-------|--------|----------|
| `src/models/Nano_U/model_tf.py` | 84-110 | Remove skip connections from call() | CRITICAL |
| `src/train_with_nas.py` | 348-390 | Add recursive layer search | CRITICAL |
| `src/train_with_nas.py` | 56, 118 | Remove inner_model from class init | HIGH |
| `src/train_with_nas.py` | 84-86, 168-170 | Remove inner_model calls in train_step | HIGH |
| `src/train_with_nas.py` | 399, 480 | Remove inner_model parameter from instantiation | HIGH |
| `src/models/Nano_U/model_tf.py` | 69 | Change MaxPool to AvgPool (optional) | LOW |

---

## Architecture Decision Summary

**Nano_U Philosophy**: Simple autoencoder (no skip connections)
- Rationale: Lighter, simpler, matches proven old implementation
- Trade-off: May have slightly lower IoU than full U-Net
- Benefit: Fewer parameters, faster inference on ESP32

**Config-Driven**: All architecture params (filters, bottleneck, decoder) can be modified via config.yaml
- Flexibility: Easy to experiment with different sizes
- Consistency: Same code works for different configurations

**NAS Integration**: Fixed to work with nested model structures
- Robustness: Handles both functional and subclassed models
- Fallback: Disables gracefully if layer detection fails

---

**Next Step**: Implement these fixes in Code mode
