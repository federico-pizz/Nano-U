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

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/Nano-U.git
cd Nano-U
python -m venv .venv && source .venv/bin/activate
pip install tensorflow numpy opencv-python PyYAML
```

### Basic Training
```bash
# Train student model with knowledge distillation
python src/train.py --model nano_u --distill --enable-nas

# Run hyperparameter experiments
python scripts/run_experiments.py --phase 4.1
```

### ESP32 Deployment
```bash
# Quantize model
python src/quantize.py --model-name nano_u --output models/nano_u_int8.tflite

# Deploy to ESP32-S3
cd esp_flash && cargo build --release
```

## 📊 Key Findings

| Metric | Teacher (BU_Net) | Student (Nano_U) | Reduction |
|--------|------------------|------------------|-----------|
| Parameters | 180K | 41K | 77% |
| Model Size | ~720KB | ~164KB | 77% |
| Quantized Size | - | ~10KB | 98.6% |
| Target Latency | - | <100ms | - |

## 🔧 Research Status

- ✅ **Architecture Design**: Depthwise separable CNN implemented
- ✅ **Training Pipeline**: Knowledge distillation with NAS monitoring
- ✅ **Quantization**: INT8 TensorFlow Lite conversion
- ⚠️ **Runtime Issues**: Model instantiation bugs (see [DEVELOPMENT.md](DEVELOPMENT.md))
- ⏳ **Hardware Validation**: ESP32-S3 benchmarking pending

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
