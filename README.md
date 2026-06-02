# Nano-U

> **3,357-parameter binary terrain segmentation for bare-metal microcontrollers.**  
> Trained via Quantization-Aware Distillation · Deployed in Rust · Runs at 1.2 FPS on a $10 SoC.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Rust](https://img.shields.io/badge/rust-esp--rs-red.svg)](https://esp-rs.github.io/book/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-TinyAgri-yellow)](https://huggingface.co/datasets/federico-pizz/TinyAgri)

---

Nano-U is a minimal encoder-decoder segmentation network designed from the ground up for autonomous navigation on commodity embedded hardware. Its architecture, training regime, and deployment stack are co-designed around a single hard constraint: **320 KB of Data RAM, no ML accelerator**.

The model is trained with **Quantization-Aware Distillation (QAD)** — a single-pass regime that fuses knowledge distillation from a full-precision BU-Net teacher with INT8 fake-quantization from epoch one — and deployed through a [fork of MicroFlow](https://github.com/federico-pizz/microflow-rs), a compile-time Rust inference engine that resolves the entire operator graph at build time with no heap allocator, no dynamic dispatch, and no interpreter overhead.

---

## Results

### Segmentation Accuracy

| Model | Dataset | mIoU | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|
| BU-Net (Teacher, Float32) | Botanic Garden | 94.6% | 95.3% | 96.6% | 93.9% |
| Nano-U (Float32) | Botanic Garden | 88.2% | 89.0% | 94.6% | 84.0% |
| **Nano-U (INT8, on ESP32)** | **Botanic Garden** | **88.1%** | **88.9%** | **94.5%** | **83.9%** |
| BU-Net (Teacher, Float32) | TinyAgri | 91.4% | 95.5% | 94.1% | 96.8% |
| Nano-U (Float32) | TinyAgri | 87.5% | 93.4% | 90.2% | 97.0% |
| **Nano-U (INT8, on ESP32)** | **TinyAgri** | **70.6%** | **83.0%** | **80.1%** | **86.1%** |

The negligible Float32→INT8 gap on Botanic Garden confirms the QAD pipeline successfully preserves accuracy through quantization. The larger drop on TinyAgri reflects the difficulty of encoding dense overlapping vegetation into INT8 precision (see [Limitations](#limitations)).

### Qualitative Results

Side-by-side comparison of raw input, ground-truth mask, and Nano-U INT8 prediction across both domains:

<img src="tools/predictions_comparison.png" alt="Input, ground truth, and Nano-U INT8 prediction on Botanic Garden and TinyAgri" width="800"/>

### Training Pipeline

The diagram below shows the QAD training loop: fake-quantization nodes are injected from epoch one so the optimizer simultaneously minimizes distillation loss (teacher soft targets) and quantization error (INT8 rounding).

<img src="tools/figure_2.png" alt="Quantization-Aware Distillation training pipeline" width="800"/>

### Hardware Efficiency (ESP32-S3-CAM)

| Metric | Value |
|:---|:---:|
| Parameters | 3,357 (INT8) |
| Model size | 33 KB |
| Peak internal RAM | 257 KB |
| Inference latency | 846 ms (~1.2 FPS) |
| Power consumption | 470 mW |

Peak RAM measured via stack painting (hardware high-water mark). Power measured at the 5V USB rail, inclusive of LDO, OV2640 camera, and USB-UART bridge.

---

## Architecture

Nano-U is a strictly sequential 7-stage encoder-decoder. Skip connections are omitted — retaining encoder feature maps in SRAM until the decoder consumes them would exceed the 320 KB budget. Depthwise separable convolutions (K=3) are used throughout.

| Stage | Spatial | Channels | Operation |
|:---|:---:|:---:|:---|
| Encoder 1 | 60×80 → 30×40 | 3 → 4 | DW-Sep Conv × 2, MaxPool 2×2 |
| Encoder 2 | 30×40 → 15×20 | 4 → 8 | DW-Sep Conv × 2, MaxPool 2×2 |
| Encoder 3 | 15×20 → 5×10 | 8 → 16 | DW-Sep Conv × 2, MaxPool **3×2** |
| Bottleneck | 5×10 | 16 | DW-Sep Conv × 2 |
| Decoder 1 | 5×10 → 15×20 | 16 → 16 | NN Upsample 3×2, DW-Sep Conv × 2 |
| Decoder 2 | 15×20 → 30×40 | 16 → 8 | NN Upsample 2×2, DW-Sep Conv × 2 |
| Decoder 3 | 30×40 → 60×80 | 8 → 1 | NN Upsample 2×2, DW-Sep Conv × 2 |

The asymmetric 3×2 MaxPool at Stage 3 reduces 15×20 cleanly to 5×10 without spatial misalignment. The largest intermediate tensor is 60×80×4 = 19.2 KB (INT8), keeping the total arena within budget.

The parameter count is reported at three stages: the plain Float32 architecture has **4,212** weights; quantization-aware training wraps these in fake-quant nodes for a saved-checkpoint count of **4,688**; and at TFLite export the batch-normalization statistics are folded into the preceding convolutions, leaving **3,357** stored INT8 parameters with no accuracy loss.

SVD-based redundancy scores (`src/nas.py`) are tracked during training to monitor layer utilization. These scores empirically confirm that encoder layers generalize across domains while decoder layers are domain-specific — motivating the decoder-only re-distillation strategy described in [Limitations](#limitations).

### Quantization-Aware Distillation

$$\mathcal{L} = \alpha \cdot T^2 \cdot \mathcal{L}_\text{KD} + (1 - \alpha) \cdot \mathcal{L}_\text{CE}$$

where $\mathcal{L}_\text{KD}$ is MSE between temperature-scaled sigmoid outputs of teacher and student, $\mathcal{L}_\text{CE}$ is binary cross-entropy against hard labels, and $T^2$ compensates for gradient magnitude reduction from temperature scaling. Fake-quantization nodes are injected from epoch one so the optimizer minimizes distillation loss and quantization error simultaneously.

---

## Datasets

Nano-U is dataset-agnostic: any binary terrain-segmentation dataset laid out as
`data/<name>/{train,val,test}/{img,mask}` works by pointing `config/config.yaml` at it.
The two datasets below are the ones used for the results in this repository and its
accompanying publication.

### Botanic Garden
An outdoor robot navigation benchmark collected in a 48,000 m² unstructured environment. We use 1,181 images from all 5 annotated sequences, split by contiguous sequence (70/20/10) to prevent temporal leakage. Binary traversability masks are derived from the original *path* class annotations.

**Original Source:** [robot-pesg/BotanicGarden](https://github.com/robot-pesg/BotanicGarden)

### TinyAgri
A custom terrain segmentation dataset collected via the onboard OV2640 camera of an ESP32-CAM mounted on a SunFounder Galaxy RVR rover. It contains 2,659 images across two agricultural environments (tomato and corn fields), annotated with SAM 2. TinyAgri is released alongside this project to support future research in edge robotics.

<img src="tools/tinyagri_grid.png" alt="Sample images from the TinyAgri dataset — tomato and corn field terrain" width="800"/>

---

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/federico-pizz/Nano-U.git
cd Nano-U
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For Rust firmware build and deployment, see [`firmware/README.md`](firmware/README.md).

---

## Usage

### Pre-trained Models

Pre-trained INT8 TFLite models for both domains are included in the repository under `models/`:

```
models/
├── BotanicGarden/nano_u.tflite
└── TinyAgri/nano_u.tflite
```

You can run evaluation directly against these without training from scratch (see [Evaluate](#evaluate) below).

### Training

Copy and fill in `config/config.yaml` with your dataset paths, then choose one of the two options below — they are mutually exclusive.

**Option A — QAD Pipeline (recommended)**

Runs all four phases in one shot: teacher training → student training with distillation → INT8 TFLite export → evaluation.

```bash
python scripts/run_qad.py --config config/config.yaml
```

**Option B — Standard training (no distillation)**

Trains teacher and student independently without knowledge distillation.

```bash
python scripts/train_model.py bu_net --config config/config.yaml   # Teacher
python scripts/train_model.py nano_u --config config/config.yaml   # Student
```

### Evaluate

```bash
python src/evaluate.py nano_u --config config/config.yaml
```

### On-Device Evaluation

```bash
python scripts/eval_esp32.py nano_u --config config/config.yaml
```

### MCU Deployment

See [`firmware/README.md`](firmware/README.md) for build setup, binary descriptions, and environment variables.

---

## Project Structure

```
Nano-U/
├── config/config.yaml          # Configuration template
├── src/
│   ├── models/                 # Nano-U and BU-Net builders
│   ├── utils/                  # Metrics, QAT wrappers, config loader
│   ├── data.py                 # tf.data pipelines and augmentation
│   ├── evaluate.py             # Evaluation and visualization
│   ├── nas.py                  # SVD-based redundancy monitoring
│   ├── quantize_model.py       # INT8 TFLite export and calibration
│   └── train.py                # Training loops (standard + distillation)
├── scripts/
│   ├── run_qad.py              # Full QAD pipeline
│   ├── train_model.py          # Single-model training
│   ├── eval_esp32.py           # On-device inference and evaluation
│   ├── profile_nano_u.py       # Stack painting and energy profiling
│   └── profile_mobilenet.py    # MobileNet baseline profiling
├── firmware/                   # ESP32-S3 bare-metal Rust
│   ├── src/bin/
│   │   ├── run.rs                    # Continuous inference loop (default target)
│   │   ├── inference.rs              # One-shot INT8 benchmark over all test images
│   │   ├── single_inference.rs       # Single-image inference + serial output
│   │   ├── analysis.rs               # Stack painting + power profiling (Nano-U)
│   │   └── analysis_person_detect.rs # Stack painting + power profiling (person_detect)
│   └── build.rs                      # Compile-time quantization param extraction + image packing
├── models/
│   ├── BotanicGarden/nano_u.tflite
│   └── TinyAgri/nano_u.tflite
└── data/
    ├── BotanicGarden/
    └── TinyAgri/
```

---

## Hardware

All experiments use an **ESP32-S3-CAM** (dual-core Xtensa LX7 @ 240 MHz, 512 KB internal SRAM, 16 MB PSRAM, OV2640 camera). No ML accelerator is present; all integer arithmetic runs on the general-purpose ALU. See [`firmware/README.md`](firmware/README.md) for build, binary descriptions, and environment variables.

---

## Limitations

- **TinyAgri quantization gap**: the 16.9 pp mIoU drop reveals that INT8 precision struggles with dense overlapping vegetation boundaries. Mixed-precision (INT16 in the final decoder stage) may help.
- **Single-core inference**: the second Xtensa LX7 core is idle. Distributing inference across both cores is the most direct path to halving latency.
- **Fixed-domain deployment**: Nano-U is currently re-trained from scratch per domain. SVD redundancy scores suggest the encoder generalizes while the decoder is domain-specific, motivating decoder-only re-distillation for new domains.

---

## Citation

If you use Nano-U or the TinyAgri dataset, please cite:

```bibtex
@misc{pizzolato2026nanou,
      title={Nano-U: Efficient Terrain Segmentation for Tiny Robot Navigation}, 
      author={Federico Pizzolato and Francesco Pasti and Nicola Bellotto},
      year={2026},
      eprint={2605.10210},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.10210}, 
}
```

---

## License

This project is dual-licensed under the **MIT License** and the **Apache License 2.0**.  
You may use it under the terms of either license, at your option.

See [LICENSE](LICENSE) for the full text of both licenses.
