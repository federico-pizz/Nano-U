# Nano-U

> Ultra-compact binary terrain segmentation for bare-metal microcontrollers, trained via Quantization-Aware Distillation and deployed through Rust.

**Nano-U** is a 3,357-parameter INT8 binary segmentation network designed for autonomous navigation in unstructured outdoor environments such as agricultural fields and natural trails. Every architectural decision is driven by the extreme constraints of commodity hardware — a general-purpose SoC costing under $10, with no ML accelerator and a strict 320 KB internal Data RAM ceiling.

The model is trained via **Quantization-Aware Distillation (QAD)**, a single-pass regime that fuses knowledge distillation from a full-scale teacher with INT8 fake-quantization from the first epoch. It is deployed through a [fork of MicroFlow](https://github.com/federico-pizz/microflow-rs), a compiler-based Rust inference engine extended with the operators required by Nano-U's architecture (`MaxPool2D` and nearest-neighbor `UpSampling2D`).

---

## Results

### Segmentation Accuracy

| Model | Dataset | mIoU | F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BU-Net (Teacher) | Botanic Garden | 96.6% | 97.1% | 97.3% | 96.8% |
| Nano-U (Float32) | Botanic Garden | 89.9% | 90.7% | 95.8% | 86.0% |
| **Nano-U (INT8, ESP32)** | **Botanic Garden** | **89.8%** | **90.7%** | **95.9%** | **86.0%** |
| BU-Net (Teacher) | TinyAgri | 92.7% | 96.2% | 94.8% | 97.6% |
| Nano-U (Float32) | TinyAgri | 88.0% | 93.6% | 92.0% | 95.3% |
| **Nano-U (INT8, ESP32)** | **TinyAgri** | **68.8%** | **81.4%** | **79.9%** | **82.9%** |

The negligible 0.1% mIoU drop from Float32 to INT8 on Botanic Garden confirms that the QAD pipeline successfully preserves accuracy through integer conversion. The larger drop on TinyAgri reflects the increased difficulty of encoding dense, overlapping vegetation boundaries into INT8 precision.

### Hardware Efficiency on ESP32-S3

| Metric | Value |
| :--- | :--- |
| Parameters | 3,357 |
| Model size (.tflite) | 33 KB |
| Output | 60×80 binary mask |
| Peak internal Data RAM | 257 KB |
| Inference latency | 844 ms (~1.2 FPS) |
| Energy per inference (SoC-level, 3.3 V) | 156 mJ |
| Energy per inference (board-level, 5 V USB) | 422 mJ |

Peak RAM is measured via **stack painting** — the stack is pre-filled with a known byte pattern before inference and scanned afterward to find the high-water mark. The 257 KB figure is a direct hardware measurement inclusive of all MicroFlow tensor allocations and call-frame overhead, and it remains safely within the 320 KB Data RAM budget.

---

## Key Design Decisions

### Architecture

Nano-U is a strictly sequential encoder-decoder with seven stages, operating on 60×80×3 input images. Skip connections are omitted: retaining encoder feature maps in RAM until the decoder consumes them would exceed the memory budget. This also ensures full compatibility with MicroFlow's static allocation model, bounding peak memory to a single active tensor buffer at any point in the inference graph.

- **Depthwise Separable Convolutions** (K=3) are used throughout, reducing both parameter count and MAC operations relative to standard convolutions.
- **Asymmetric Pooling**: a 3×2 MaxPool at Stage 3 reduces 15×20 cleanly to 5×10, avoiding the spatial misalignment a standard 2×2 pool would introduce.
- **Nearest-Neighbor Upsampling** in the decoder requires no multiply-accumulate operations and is numerically stable under INT8 quantization.
- **Channel widths** [4, 8, 16] are co-designed with spatial dimensions so the largest intermediate activation tensor is 60×80×4 = 19.2 KB in INT8, keeping the total tensor arena within 320 KB.

| Stage | Input | Output | Channels | Op |
| :--- | :--- | :--- | :--- | :--- |
| Encoder 1 | 60×80 | 30×40 | 3→4 | MaxPool 2×2 |
| Encoder 2 | 30×40 | 15×20 | 4→8 | MaxPool 2×2 |
| Encoder 3 | 15×20 | 5×10 | 8→16 | MaxPool 3×2 |
| Bottleneck | 5×10 | 5×10 | 16→16 | DW-Sep Conv |
| Decoder 1 | 5×10 | 15×20 | 16→8 | NN 3×2 |
| Decoder 2 | 15×20 | 30×40 | 8→4 | NN 2×2 |
| Decoder 3 | 30×40 | 60×80 | 4→1 | NN 2×2 |

### Quantization-Aware Distillation (QAD)

Training Nano-U presents two concurrent challenges: its minimal parameter budget limits representational capacity when trained on hard labels alone, and INT8 deployment introduces accuracy degradation if quantization is not accounted for during training.

QAD addresses both simultaneously in a single optimization pass from epoch one:

- **Knowledge Distillation**: soft targets from BU-Net (12.85M parameters) guide the student via a temperature-scaled sigmoid loss. Temperature T=8.0 and α=0.3 are used for Botanic Garden; T=4.0 and α=0.5 for TinyAgri.
- **Quantization-Aware Training (QAT)**: fake-quantization nodes injected from the first epoch allow the optimizer to simultaneously minimize the distillation loss and quantization error, preventing the student from converging to float32 weights that are subsequently perturbed by integer conversion.

The total loss is:

$$\mathcal{L} = \alpha \cdot T^2 \cdot \mathcal{L}_\text{KD} + (1 - \alpha) \cdot \mathcal{L}_\text{CE}$$

where $\mathcal{L}_\text{KD}$ is the MSE between temperature-scaled sigmoid outputs of teacher and student, $\mathcal{L}_\text{CE}$ is binary cross-entropy against the hard ground-truth mask, and the $T^2$ factor compensates for the reduction in gradient magnitude induced by temperature scaling.

### Rust Deployment via MicroFlow

Inference runs on bare-metal Xtensa LX7 using our [fork of MicroFlow](https://github.com/federico-pizz/microflow-rs), extended with three operators required by Nano-U: `MaxPool2D`, nearest-neighbor `UpSampling2D`, and `Sigmoid`.

- **Compile-Time Graph Evaluation**: MicroFlow resolves the full operator graph at compile time via a Rust procedural macro. All tensor buffers are placed in static DRAM at link time — no heap allocator, no dynamic dispatch, no interpreter overhead.
- **`no_std` environment**: the wireless stack is never initialized. Only the CPU, internal DRAM, and PSRAM (for input image storage) are active during inference. This minimalist execution model is responsible for the low energy figures reported above.

```rust
#[model("models/nano_u.tflite")]
struct UNet;

// Inference
let mask = UNet::predict_quantized(input_batch);
```

---

## Datasets

### Botanic Garden

A comprehensive outdoor robot navigation benchmark collected in a 48,000 m² unstructured environment. We use 1,181 images from all 5 annotated sequences. To prevent temporal data leakage, data is split by contiguous sequence (70/20/10 train/val/test) rather than random frame sampling. Binary traversability masks are extracted from the original *path* class annotations.

### TinyAgri

A custom terrain segmentation dataset collected via the onboard OV2640 camera of an ESP32-CAM module mounted on a SunFounder Galaxy RVR rover. It contains 2,659 images across two agricultural environments (tomato and corn fields), annotated using SAM 2. TinyAgri is publicly released alongside this project to support future research in edge robotics.

TinyAgri is significantly harder than Botanic Garden: dense foliage, greater leaf overlap, motion blur, and the inherently lower image quality of the onboard camera all contribute to a more challenging segmentation problem that tests cross-domain generalization.

---

## Project Structure

```
Nano-U/
├── config/
│   ├── config.yaml                  # Default configuration (Botanic Garden)
│   ├── botanic_garden_config.yaml   # Botanic Garden dataset configuration
│   └── tinyagri_config.yaml         # TinyAgri dataset configuration
├── src/
│   ├── models/                      # Nano-U and BU-Net model builders
│   ├── utils/                       # QAT wrappers, config loader, metrics
│   ├── data.py                      # tf.data pipelines and augmentation
│   ├── evaluate.py                  # Evaluation and visualization tools
│   ├── nas.py                       # SVD-based redundancy monitoring (NASCallback)
│   ├── pipeline.py                  # Training orchestrators
│   ├── quantize_model.py            # INT8 TFLite export and calibration
│   └── train.py                     # Training loops (standard and distillation)
├── scripts/
│   ├── train_standard.py            # Standard training without distillation
│   ├── train_distillation.py        # Full QAD pipeline (teacher → student)
│   ├── benchmark_on_esp.py          # On-device inference and metric evaluation
│   └── stack_analyzer.py            # On-device stack and energy profiling
├── esp_flash/                       # ESP32-S3 bare-metal Rust firmware
│   ├── src/bin/
│   │   ├── main.rs                  # Float32 inference loop
│   │   ├── inference.rs             # INT8 inference with full output capture
│   │   ├── single_inference.rs      # Single-image INT8 inference
│   │   └── analysis.rs              # Stack painting and energy profiling
│   ├── build.rs                     # Compile-time quantization param extraction
│   └── Cargo.toml
├── results/                         # Training logs, metrics, and plots
└── tests/                           # Pytest suite
```

---

## Installation

Requires Python 3.12+.

```bash
python -m venv nano_u_venv
source nano_u_venv/bin/activate
pip install -r requirements.txt
```

For the Rust firmware, install the ESP Rust toolchain:

```bash
# Follow https://esp-rs.github.io/book/installation/index.html
source $HOME/export-esp.sh
```

---

## Training

### Full QAD Pipeline (recommended)

Trains BU-Net to convergence, then trains Nano-U via Quantization-Aware Distillation, exports INT8 TFLite, and runs benchmarks:

```bash
# Botanic Garden (default)
python scripts/train_distillation.py

# TinyAgri
python scripts/train_distillation.py --config config/tinyagri_config.yaml
```

### Standard Training (no distillation)

Trains a single model without knowledge distillation. Useful for training BU-Net as a standalone teacher or for ablation:

```bash
# Train the BU-Net teacher
python scripts/train_standard.py bu_net

# Train Nano-U without distillation (weaker baseline)
python scripts/train_standard.py nano_u --config config/botanic_garden_config.yaml
```

### Manual INT8 Export

To export an existing `.keras` model to a calibrated INT8 `.tflite` file:

```bash
python src/quantize_model.py path/to/model.keras path/to/model.tflite --config config/botanic_garden_config.yaml
```

Calibration uses the full validation split to compute INT8 scale factors and zero-points. Quantization parameters are also saved to a companion `_quant_params.json` file consumed by the Rust build script.

---

## Evaluation

Evaluate segmentation metrics on the test split and generate prediction visualizations:

```bash
python src/evaluate.py nano_u --config config/botanic_garden_config.yaml
```

The script automatically resolves the best available model format (`.tflite` > `.h5`). Results and plots are saved to `results/`.

---

## MCU Deployment

### Flashing and Running Inference

```bash
cd esp_flash
source $HOME/export-esp.sh

# Standard INT8 inference loop
cargo run --release --bin inference

# Single-image inference with full output capture
cargo run --release --bin single_inference
```

To target a different model or dataset directory:

```bash
MODEL_NAME=nano_u \
MODELS_DIR=models/tinyagri \
TEST_IMG_DIR=data/tinyagri/test/img \
cargo run --release --bin inference
```

### On-Device Evaluation

Run inference on the ESP32 and compute segmentation metrics against the test set ground truth:

```bash
python scripts/benchmark_on_esp.py nano_u --config config/botanic_garden_config.yaml
```

### Hardware Profiling (Stack and Energy)

Measures peak RAM consumption via stack painting and provides a steady-state inference loop for current measurement with a multimeter:

```bash
# Requires ESP32-S3 connected via USB
python scripts/stack_analyzer.py
```

Results and plots are saved to `results/nano_u/`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Hardware Notes

All on-device experiments use an **ESP32-S3-CAM** (dual-core Xtensa LX7 @ 240 MHz, 512 KB internal SRAM, 16 MB PSRAM, OV2640 camera). The SoC provides ~320 KB of Data RAM and ~192 KB of Instruction RAM. No ML accelerator is present; all integer arithmetic executes on the general-purpose ALU.

During benchmarks, the OV2640 camera is not used for live capture. Input images are loaded from PSRAM to isolate pure inference latency from peripheral I/O. Inference is constrained to a single core, leaving the secondary core available for concurrent system tasks.

SoC-level energy (156 mJ) is measured by reading current directly at the 3.3 V pin. Board-level energy (422 mJ) is measured at the 5 V USB supply rail and includes the LDO regulator, camera module, and USB-UART bridge.

---

## Limitations and Future Work

- **Quantization bottleneck on complex scenes**: the 68.8% mIoU on TinyAgri reveals that forcing dense, overlapping vegetation boundaries into INT8 precision causes accuracy degradation the QAD pipeline only partially mitigates. Higher bit-width quantization or mixed-precision schemes may help.
- **Single-core inference**: the secondary Xtensa LX7 core is currently idle during inference. Distributing the inference graph across both cores within MicroFlow is the most direct path to reducing the 844 ms latency.
- **Board-level energy measurement**: reported figures are inclusive of the LDO, camera module, and USB-UART bridge. GPIO-isolated measurement at the SoC power pin would yield a more precise per-inference figure.
- **Fixed-domain deployment**: Nano-U is currently re-trained from scratch for each new domain. On-device continual learning using the second core could enable deployment-time adaptation to unseen terrain types without full retraining.

---

## Citation

If you use Nano-U or the TinyAgri dataset in your work, please cite:

```bibtex
@inproceedings{pizzolato2025nanou,
  title     = {Nano-U: Efficient Terrain Segmentation for Tiny Robot Navigation},
  author    = {Pizzolato, Federico and Pasti, Francesco and Bellotto, Nicola},
  booktitle = {TODO: conference/journal},
  year      = {2025}
}
```

---

## License

See [LICENSE](LICENSE) for details.