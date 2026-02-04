# Nano-U: Ultra-Low-Power CNN for Microcontroller-Based Real-Time Segmentation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)
![TensorFlow 2.21+](https://img.shields.io/badge/tensorflow-2.21+-blue)
![Research](https://img.shields.io/badge/status-research-orange)

> **Research Goal**: Demonstrate real-time semantic segmentation for autonomous navigation on energy-constrained microcontrollers (ESP32-S3) with <100ms latency and <1W power consumption.

## 🔬 Core Methodology

**Nano-U** investigates extreme CNN miniaturization through:
- **Knowledge Distillation**: 180K→41K parameter compression (77% reduction)
- **Depthwise Separable Architecture**: Optimized for microcontroller constraints
- **Real-time NAS Monitoring**: Live covariance analysis for redundancy detection
- **INT8 Quantization**: ~10KB deployment models for ESP32-S3

### Architecture Overview

```
Teacher (BU_Net): 180K params, U-Net with skip connections
Student (Nano_U): 41K params, pure autoencoder design
Compression: 77% parameter reduction
Target: ESP32-S3 (520KB SRAM, 8MB PSRAM)
```

### Key Research Contributions

1. **Extreme Compression Pipeline**: Complete framework for <50K parameter segmentation
2. **Live NAS Integration**: Novel real-time redundancy monitoring during training
3. **Microcontroller Deployment**: End-to-end ESP32-S3 inference pipeline

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/Nano-U.git
cd Nano-U
python -m venv .venv-tf && source .venv-tf/bin/activate  # or .venv on Windows
pip install -r requirements.txt
```

### Training (single experiment)
```bash
# From config: run one experiment by name (e.g. quick_test, standard, distillation)
python src/train.py --config config/experiments.yaml --experiment quick_test --output results/
```

### Running experiments (recommended)
```bash
# List experiments
python scripts/run_experiments.py --config config/experiments.yaml --list

# Run one experiment
python scripts/run_experiments.py --experiment quick_test --output results/
```

### Entry points (code)
- **Models**: `from src.models import create_nano_u, create_bu_net, create_model_from_config`
- **Training**: `from src.train import train_model, train_single_model, train_with_distillation`
- **Experiments**: `scripts/run_experiments.py` is the single entry point to run experiments; it loads config, creates an output dir, and calls `train_model`. No separate `src/experiment.py` — see [REFACTURING_DOCUMENTATION.md](REFACTURING_DOCUMENTATION.md) for API and migration.
- **Config**: `config/experiments.yaml` — one file for all experiment settings

### Testing
```bash
# From project root (with venv activated)
pytest tests/ -v
# Integration tests (pipeline, NAS, model size): tests/test_pipeline.py
```

## 📊 Key Findings

| Metric | Teacher (BU_Net) | Student (Nano_U) | Reduction |
|--------|------------------|------------------|-----------|
| Parameters | 180K | 41K | 77% |
| Model Size | ~720KB | ~164KB | 77% |
| Quantized Size | - | ~10KB | 98.6% |
| Target Latency | - | <100ms | - |

## Research Status

- **Architecture**: Depthwise separable U-Net (Functional API) in `src/models/`
- **Training**: Unified pipeline in `src/train.py`; experiments via `scripts/run_experiments.py`
- **NAS**: Lightweight callback in `src/nas.py` (redundancy metrics per epoch)
- **Config**: Single `config/experiments.yaml`; old configs: `python scripts/migrate_config.py old.yaml new.yaml`

## 📚 Documentation

- [**RESEARCH_PAPER.md**](RESEARCH_PAPER.md) - Detailed methodology and experimental results
- [**API_REFERENCE.md**](API_REFERENCE.md) - Code documentation and usage examples
- [**INSTALLATION.md**](INSTALLATION.md) - Setup instructions and dependencies
- [**DEVELOPMENT.md**](DEVELOPMENT.md) - Research roadmap and optimization targets

## 🎯 Research Applications

- **Autonomous Navigation**: Vision-based path planning on drones/robots
- **Agricultural Robotics**: Crop/weed segmentation for precision farming
- **Edge IoT**: Ultra-low-power computer vision applications

## 📖 Citation

```bibtex
add here when it will be ready
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

**Research Contact**: [your.email@institution.edu]  
**Last Updated**: 2026-02-04  
**Status**: Active Research Project
