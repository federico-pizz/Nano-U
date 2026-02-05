# Nano-U: Ultra-Low-Power CNN for Microcontroller-Based Real-Time Segmentation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)
![TensorFlow 2.21+](https://img.shields.io/badge/tensorflow-2.21+-blue)
![Research](https://img.shields.io/badge/status-research-orange)

> **Research Goal**: Demonstrate real-time semantic segmentation for autonomous navigation on energy-constrained microcontrollers (ESP32-S3) with <100ms latency and <1W power consumption.

---

## 🔬 Project Overview

**Nano-U** investigates extreme CNN miniaturization for edge robotics:
- **Knowledge Distillation**: 180K→41K parameter compression (77% reduction).
- **Depthwise Separable Architecture**: Optimized for microcontroller memory constraints.
- **Real-time NAS Monitoring**: Live SVD-based redundancy analysis for layer optimization.
- **Microflow Compatibility**: Models designed for Rust-based inference engines.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/Nano-U.git
cd Nano-U
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Automated Pipeline (Recommended)
Run the full Training -> Distillation -> Benchmarking flow:
```bash
python scripts/run_pipeline.py --full
```

### Individual Experiments
```bash
# List available experiments in config/experiments.yaml
python scripts/run_pipeline.py --list

# Run a specific experiment
python scripts/run_pipeline.py --experiment quick_test
```

### Evaluation
```bash
python src/evaluate.py --model-name nano_u
```

## 📊 Key Results

| Metric | Teacher (BU_Net) | Student (Nano_U) | Reduction |
|--------|------------------|------------------|-----------|
| Parameters | 180K | 41K | 77% |
| Model Size | ~720KB | ~164KB | 77% |
| Quantized Size | - | ~10KB | 98.6% |
| Target Latency | - | <100ms | - |

## 📚 Documentation

- [**API_REFERENCE.md**](API_REFERENCE.md) - Code documentation and CLI usage.
- [**DEVELOPMENT.md**](DEVELOPMENT.md) - Research roadmap and current objectives.
- [**RESEARCH_PAPER.md**](RESEARCH_PAPER.md) - Detailed methodology and experimental results.

---

**Last Updated**: 2026-02-05  
**Status**: Active Research Project
