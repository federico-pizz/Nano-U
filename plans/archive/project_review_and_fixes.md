# Nano-U Project Review & Simplification Plan

## Executive Summary

The project has become overcomplicated during development with architectural mismatches between the old working PyTorch implementation and the new TensorFlow implementation. Key issues include:

1. **Nano_U architecture changed**: Added skip connections (U-Net style) when old version had none
2. **train_with_nas.py has bugs**: Layer detection fails with nested functional/subclassed model structure
3. **Config misalignment**: Parameters don't match the original working implementation
4. **Unnecessary complexity**: Functional wrappers around subclassed models make NAS integration difficult

---

## Critical Findings

### 1. Nano_U Architecture Mismatch

#### Old Implementation (PyTorch - WORKING)
```
Architecture: Simple autoencoder WITHOUT skip connections
- Encoder: [32, 64, 128] with AvgPool2d downsampling
- Bottleneck: 128 filters
- Decoder: [64, 32] with bilinear upsampling (NO skip connections)
- Convolutions: DoubleConv (2x depthwise separable conv blocks)
- Total: ~41K parameters
```

#### New Implementation (TensorFlow - PROBLEMATIC)
```
Architecture: U-Net WITH skip connections
- Encoder: [16, 32, 64] with MaxPool2d downsampling  
- Bottleneck: 64 filters
- Decoder: [32, 16] with skip connections from encoder
- Convolutions: DepthwiseSepConv (depthwise separable)
- Total: ~19K parameters (too small!)
```

**Issue**: The new implementation is trying to be a U-Net but the old was simpler. Skip connections were added during development, changing the fundamental architecture.

**Solution**: Remove skip connections from Nano_U to match old implementation OR keep them but adjust filter sizes to maintain parameter count.

---

### 2. BU_Net Architecture Review

#### Old Implementation (PyTorch)
```
- Encoder: [64, 128, 256, 512, 1024, 2048] (6 levels)
- Bottleneck: 2048 filters
- Decoder: [1024, 512, 256, 128, 64] (5 upsampling stages)
- Convolutions: TripleConv (3x depthwise separable conv blocks)
- Skip connections: YES (proper U-Net)
- Total: ~2.8M parameters
```

#### New Implementation (TensorFlow)
```
- Architecture matches old implementation correctly ✓
- Uses DepthwiseConv2D properly ✓
- Skip connections implemented correctly ✓
```

**Status**: BU_Net implementation is CORRECT - no changes needed.

---

### 3. train_with_nas.py Critical Bugs

#### Bug #1: Nested Model Detection (Lines 356-361)
```python
# Current code tries to find nested subclassed model
for layer in student_model.layers:
    if isinstance(layer, tf.keras.Model) and layer.name in ['BU_Net', 'Nano_U']:
        inner_model = layer
        break
```

**Problem**: 
- `build_nano_u()` returns a functional Model wrapping a NanoU subclassed instance
- The subclassed instance IS in model.layers, but ActivationExtractor needs direct access
- When inner_model is detected, it's passed to ActivationExtractor, but the extractor still uses the wrong model

#### Bug #2: ActivationExtractor Initialization (Line 385)
```python
model_for_extractor = inner_model if inner_model is not None else student_model
extractor = ActivationExtractor(model_for_extractor, selectors)
```

**Problem**:
- If inner_model is the subclassed NanoU, it doesn't have all intermediate layers accessible
- The functional wrapper has better layer visibility
- But auto-detection fails because layer names don't match expected patterns

#### Bug #3: Silent Failure
When ActivationExtractor fails to find layers, it falls back but training continues without NAS, making debugging difficult.

---

### 4. Configuration Issues

#### Current config.yaml
```yaml
nano_u:
  filters: [16, 32, 64]        # Too small
  bottleneck: 64               # Matches new but not old
  decoder_filters: [32, 16]    # New style
```

#### Should be (to match old)
```yaml
nano_u:
  filters: [32, 64, 128]       # Match old encoder
  bottleneck: 128              # Match old bottleneck
  decoder_filters: [64, 32]    # Match old decoder
```

---

## Root Cause Analysis

The project complexity grew because:

1. **Architecture evolution**: Someone tried to make Nano_U more sophisticated (U-Net with skip connections) without updating documentation
2. **NAS integration difficulty**: The functional wrapper pattern makes layer access complex
3. **Config drift**: Parameters were reduced to make model "lighter" but this changed the architecture significantly
4. **Lack of validation**: No tests comparing old vs new parameter counts

---

## Recommended Solutions (Priority Order)

### Priority 1: Fix Nano_U Architecture
**Decision needed**: Choose ONE approach:

**Option A: Match Old Implementation (Recommended)**
- Remove skip connections from Nano_U
- Change filters to [32, 64, 128]
- Change bottleneck to 128
- Change decoder to [64, 32]
- Use DoubleConv (2 depthwise separable blocks) instead of single DepthwiseSepConv
- Target ~41K parameters

**Option B: Keep New But Fix**
- Keep skip connections
- Increase filters to compensate: [24, 48, 96]
- Increase bottleneck to 96
- Adjust decoder: [48, 24]
- Target ~41K parameters

### Priority 2: Fix train_with_nas.py

**Solution 1: Simplify Model Building**
Return subclassed models directly when NAS is enabled:
```python
def build_nano_u(..., for_nas=False):
    if for_nas:
        # Return subclassed model directly
        return NanoU(...)
    else:
        # Return functional wrapper for better Keras compatibility
        return functional_wrapper(NanoU(...))
```

**Solution 2: Fix Layer Detection**
Improve pattern matching to work with both functional and subclassed models:
```python
# Better pattern: look for any conv layer, not specific names
all_conv = UNetLayerSelector.get_all_conv_layers(student_model)
# Then filter by pattern if needed
encoder_conv = [l for l in all_conv if 'encoder' in l or 'down' in l]
```

### Priority 3: Update Config
Match old implementation parameters in config.yaml

### Priority 4: Add Validation
Create test to ensure architectures match expected parameter counts

---

## Detailed Action Plan

### Phase 1: Architecture Alignment
1. Review old Nano_U implementation and count parameters
2. Decide on Option A (match old) or Option B (keep new but fix)
3. Update `src/models/Nano_U/model_tf.py` accordingly
4. Update `config/config.yaml` with correct parameters
5. Create test script to verify parameter counts

### Phase 2: Fix NAS Integration
1. Analyze layer naming in both models
2. Update `train_with_nas.py` layer detection logic
3. Test ActivationExtractor with both functional and subclassed models
4. Add better error messages when layers aren't found
5. Consider adding `for_nas` flag to build functions

### Phase 3: Testing & Validation
1. Test training WITHOUT NAS (baseline)
2. Test training WITH NAS
3. Compare results to old implementation
4. Verify quantization pipeline works
5. Test full pipeline with scripts/tf_pipeline.sh

### Phase 4: Simplification & Documentation
1. Remove unused/redundant code
2. Update README with architecture decisions
3. Document why skip connections were removed (if Option A)
4. Add architecture diagrams
5. Create comparison table: old vs new

---

## Architecture Comparison Diagrams

### Old Nano_U (Simple Autoencoder)
```
Input (48x64x3)
    ↓
[DoubleConv 32] (48x64x32)
    ↓ AvgPool
[DoubleConv 64] (24x32x64)
    ↓ AvgPool
[DoubleConv 128] (12x16x128)
    ↓
[DoubleConv 128] Bottleneck (12x16x128)
    ↓ Upsample
[DoubleConv 64] (24x32x64)
    ↓ Upsample
[DoubleConv 32] (48x64x32)
    ↓
[Conv 1x1] Output (48x64x1)
```

### New Nano_U (U-Net with Skip Connections)
```
Input (48x64x3)
    ↓
[DepthSepConv 16] ─────┐ skip1
    ↓ MaxPool           │
[DepthSepConv 32] ────┐│ skip2
    ↓ MaxPool          ││
[DepthSepConv 64]      ││
    ↓                  ││
[DepthSepConv 64] Bottleneck
    ↓ Upsample         ││
[Concat skip2] ────────┘│
[DepthSepConv 32]       │
    ↓ Upsample          │
[Concat skip1] ─────────┘
[DepthSepConv 16]
    ↓
[Conv 1x1] Output
```

**Key Difference**: Old = no skip connections (simpler), New = with skip connections (U-Net)

---

## Questions for Clarification

1. **Architecture Philosophy**: Should Nano_U be a simple autoencoder (like old) or a proper U-Net (current)?

2. **Parameter Budget**: The old Nano_U had ~41K params. Should we match this or can we use fewer?

3. **NAS Priority**: Is NAS functionality critical, or can we deprioritize it to get basic training working first?

4. **Skip Connections**: Were skip connections added intentionally for better performance, or accidentally during refactoring?

5. **Pooling Type**: Old used AvgPool2d, new uses MaxPool2d. Does this matter?

---

## Expected Outcomes After Fixes

1. ✓ Nano_U architecture matches old implementation
2. ✓ Parameter count ~41K (matches old)
3. ✓ Training works without NAS
4. ✓ Training works with NAS (if needed)
5. ✓ Quantization pipeline works
6. ✓ Model performance matches or exceeds old implementation
7. ✓ Code is simpler and easier to maintain

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Architecture change breaks performance | Medium | High | Test thoroughly before committing |
| NAS still doesn't work after fixes | Medium | Medium | Make NAS optional, focus on baseline |
| Parameter count doesn't match | Low | Medium | Calculate before implementing |
| Training time increases | Low | Low | Monitor and adjust if needed |
| Quantization fails | Low | High | Test early in process |

---

## Next Steps

1. **User Decision Required**: Choose Option A (match old) or Option B (improve new)
2. **Implementation**: Execute fixes based on chosen option
3. **Testing**: Validate against old implementation results
4. **Documentation**: Update all docs to reflect changes

---

**Document Version**: 1.0  
**Date**: 2026-01-16  
**Reviewer**: Architect Mode
