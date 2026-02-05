# Development Roadmap: Ultra-Low-Power Research

## 🧬 Current Status

The initial refactoring phase is **Complete**. The codebase has been transitioned from a complex, non-functional state to a streamlined research framework with:
- **90%+ Experiment Success Rate**: Functional API models with robust serialization.
- **Stable NAS Metrics**: SVD-based redundancy computation.
- **Unified Pipeline**: End-to-end automation from training to benchmarking.
- **Microflow Compatibility**: Architectures optimized for Rust-based inference engines (no skip connections, supported ops only).

---

## 🎯 Active Research Objectives

### 1. Hardware-in-the-Loop Validation (ESP32-S3)
- [ ] Deploy INT8 quantized Nano-U models to ESP32-S3.
- [ ] Measure real-world latency (Target: <100ms).
- [ ] Profile power consumption during continuous inference (Target: <1W).

### 2. TinyAgri Dataset Generalization
- [ ] Validate performance on Tomatoes and Crops subsets.
- [ ] Implement multi-spectral data fusion (RGB + NIR) if available.
- [ ] Fine-tune distillation temperature for agricultural texture segmentation.

### 3. Evolutionary Neural Architecture Search (NAS)
- [ ] Expand NAS search space beyond filter counts (block types, kernel sizes).
- [ ] Implement multi-objective NAS (Accuracy vs Latency vs Energy).
- [ ] Integrate Microflow-specific constraints into the fitness function.

---

## 📈 Recent Milestones

| Milestone | Date | Status | Result |
|-----------|------|--------|--------|
| Refactoring Phase 1 | 2026-02-04 | ✅ | Code reduced by 70%, 100% stability. |
| Microflow Optimization | 2026-02-05 | ✅ | Successful conversion of models for Rust engine. |
| End-to-End Pipeline | 2026-02-05 | ✅ | Fully automated Training -> Distill -> Quant -> Bench. |

---

## 🛠️ Next Technical Priorities

### Immediate
- **Deployment**: Finalize `esp_flash` integration with Microflow.
- **NAS Stability**: Refine covariance normalization for very small feature maps.
- **Documentation**: Maintain single source of truth for experiment configurations.

### Short-Term (Months 1-2)
- **Paper Draft**: Compile benchmarks for teacher vs student vs quantized student.
- **Field Testing**: Baseline data collection in greenhouse environments.
- **Optimization**: Explore pruning as a post-distillation compression step.

## 🚧 Risk Mitigation

- **Constraint Drifts**: Regularly run `tests/test_microflow_compatibility.py` to ensure models remain deployable.
- **Numerical Drift**: Monitor SVD condition numbers during long NAS runs.
- **Hardware Limitations**: Focus on INT8 optimization to stay within ESP32-S3 SRAM limits (320KB).

---

*Last Updated: 2026-02-05*
