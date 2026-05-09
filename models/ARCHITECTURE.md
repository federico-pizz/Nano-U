# Nano-U Architecture Report: Efficient Edge Segmentation

This document breaks down the core architectural design of the Nano-U machine learning pipeline. The project is built to solve binary image segmentation tasks using a heavy-to-light knowledge distillation approach, specifically targeting extremely constrained devices like the ESP32 microcontroller which provides only ~300 KB of usable SRAM.

## 1. The Big Picture: Heavy Teacher, Ultra-Light Student

Deploying deep learning on edge devices requires operating within extreme hardware limits. The ESP32 imposes a strict ceiling on the intermediate memory (the BSS arena) required during inference. To circumvent this without sacrificing accuracy, the framework utilizes a dual-model setup: 
- **`BU_Net`**, a massive "teacher" model that runs purely offline to learn the dataset.
- **`Nano_U`**, an ultra-light "student" model that runs natively on the ESP32 and learns directly from the teacher.

## 2. The Teacher: BU_Net

`BU_Net` is built on the classic U-Net design. It acts as a high-capacity oracle during training.

### The Computational Profile
- **Depth and Width**: It uses 5 max-pooling stages to compress a $60 \times 80$ image down to $3 \times 5$, while aggressively widening the channels at each step: `[64, 128, 256, 512, 1024]`.
- **Skip Connections**: It exploits U-Net's trademark skip connections to pass high-resolution edge details directly across the network.
- **The Cost of Standard Convolutions**: A normal convolution applied to a spatial grid of $H \times W$ with $C_{in}$ input channels, $C_{out}$ output channels, and a $K \times K$ kernel costs:
  - **Parameters**: $K^2 \cdot C_{in} \cdot C_{out}$
  - **MACs (Multiply-Accumulates)**: $H \cdot W \cdot K^2 \cdot C_{in} \cdot C_{out}$

Because this cost scales quadratically with channel width, the bottleneck layer alone contains over a million parameters. It is mathematically untenable to run this on an ESP32, but it is structurally perfect for teaching.

## 3. The Student: Nano_U (Built for ESP32)

To allow `Nano_U` to execute within the ESP32's ~300 KB BSS arena via `microflow-rs` and TFLite Micro, the architecture radically departs from the standard U-Net formula.

### Sequential Topology (No Skip Connections)
`Nano_U` explicitly omits skip connections. If skip connections are used, the inference engine is forced to hold massive, high-resolution tensors in static RAM until the decoder is ready to concatenate them later. By designing the network to be strictly sequential, `microflow-rs` can immediately overwrite old tensors in the memory arena, drastically slashing peak SRAM usage.

### Depthwise Separable Convolutions (The 8x Savings)
All standard convolutions are replaced with Depthwise Separable Convolutions. This factorizes the computation into two cheaper steps:
1. **Depthwise**: A spatial filter ($K \times K$) applied independently per channel. (Cost: $H \cdot W \cdot K^2 \cdot C_{in}$ MACs).
2. **Pointwise**: A $1 \times 1$ convolution across all channels. (Cost: $H \cdot W \cdot C_{in} \cdot C_{out}$ MACs).

**The Mathematical Advantage**: 
The computational ratio between a depthwise separable convolution and a standard convolution is:
$$ \frac{1}{C_{out}} + \frac{1}{K^2} $$
For a $3 \times 3$ kernel ($K=3$), this technique natively cuts total parameters and MACs by approximately **~8-9x**.

### Compound Scaling: Designing for the Arena
Fitting a network into an MCU requires precise mathematical balancing of resolution, depth, and width—a process called compound scaling.
- **Fixed Input**: The input resolution is locked to $60 \times 80 \times 3$ (14.4 KB).
- **The BSS Bottleneck**: Processing large spatial grids early in a network causes fatal memory faults. For instance, a single $60 \times 80 \times 16$ tensor requires 76.8 KB. In depthwise convolutions, multiple such buffers must exist simultaneously, which would instantly overflow the 300 KB limit.
- **The Mathematical Solution**: `Nano_U` is designed with an aggressive 3-stage downsampling topology and highly constrained channel widths `[4, 8, 16]`:
  - **Stage 1**: $60 \times 80 \rightarrow 30 \times 40$ (pool $2 \times 2$)
  - **Stage 2**: $30 \times 40 \rightarrow 15 \times 20$ (pool $2 \times 2$)
  - **Stage 3**: $15 \times 20 \rightarrow 5 \times 10$ (**custom pool $3 \times 2$**)

By utilizing a meticulously chosen $3 \times 2$ pool to crush the spatial map down to $5 \times 10$ before scaling up to 16 channels, the architecture mathematically bounds its largest intermediate tensor (which occurs early on with just 4 channels) to exactly $60 \cdot 80 \cdot 4 = 19,200$ bytes (**19.2 KB**). This completely neutralizes the memory fault, keeping the entire dynamic arena footprint safely within the ESP32 constraints.

## 4. Heavy-to-Light Knowledge Distillation (KD)

An ultra-light model like `Nano_U` (~6,036 parameters) lacks the representational capacity to successfully generalize if trained purely on hard ground-truth labels. 

Instead, the framework relies on Knowledge Distillation. `Nano_U` is trained to mimic the soft probability distributions (logits) produced by the massive `BU_Net` teacher.
- **Temperature Scaling ($T$)**: The softmax outputs are softened during training using a temperature variable $T$:
  $$ q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} $$
  By setting $T=4.0$, the distribution flattens. Rather than just learning rigid binary class boundaries, the student absorbs the nuanced relative similarities between classes (the "dark knowledge").
- **The Synergy with Compound Scaling**: Knowledge distillation only succeeds if the student possesses a structurally balanced capacity to absorb the knowledge. Blindly trimming channels from a larger model destroys logical pathways. Because `Nano_U` is deliberately architected using compound scaling—with a specific $3 \times 2$ depth progression and $[4, 8, 16]$ channel constraints—it preserves sufficient structural feature diversity deep in the network. This compound balancing provides the exact mathematical canvas necessary for distillation to actually take root.

## 5. Experimental Setup & Optimization

To ensure reproducibility, the research follows a strict hyperparameter regime as defined in the global configuration.

### Hyperparameter Configuration

| Parameter | Teacher (BU_Net) | Student (Nano_U) |
| :--- | :--- | :--- |
| **Epochs** | 100 | 100 |
| **Batch Size** | 32 | 16 |
| **Optimizer** | AdamW | AdamW |
| **Learning Rate** | $1 \cdot 10^{-3}$ | $1 \cdot 10^{-3}$ |
| **Weight Decay** | $1 \cdot 10^{-3}$ | $1 \cdot 10^{-4}$ |
| **Patience** | 20 | 20 |
| **Scheduler** | ReduceLROnPlateau | ReduceLROnPlateau |
| **Distillation $\alpha$** | N/A | 0.5 |
| **Distillation $T$** | N/A | 4.0 |

### Dataset Pipeline
- **Input Resolution**: $60 \times 80 \times 3$ (RGB)
- **Normalization**: Zero-mean, unit-variance ($[0.5, 0.5, 0.5]$ offsets)
- **Augmentation**: Random horizontal flip, $\pm 20^\circ$ rotation, and color jitter (brightness/contrast/saturation/hue).

## 6. Quantization-Aware Training (QAT)

To minimize the accuracy gap between training in float32 and inference in INT8, `Nano_U` utilizes **Quantization-Aware Training (QAT)** via the TensorFlow Model Optimization (TFMOT) toolkit.

- **Annotate-then-Apply Pattern**: The pipeline uses a custom `apply_qat_to_model` utility that handles layers TFMOT cannot natively quantize (e.g., `UpSampling2D`, `MaxPooling2D`, `Concatenate`) using a `NoOpQuantizeConfig`. This ensures the structure remains compatible with the Rust inference engine.
- **Fake Quantization Nodes**: During training, QAT injects "fake quantization" nodes into the graph. These nodes simulate the rounding and clamping effects of 8-bit quantization, forcing the optimizer to find weights that are robust to integer precision.
- **On-Device Accuracy**: This methodology significantly improves IoU on the ESP32 compared to traditional Post-Training Quantization (PTQ).

## 7. Knowledge Retention & Data Rehearsal

When fine-tuning the model on new domains (e.g., moving from `BotanicGarden` to `TinyAgri`), `Nano_U` employs a **Data Rehearsal** strategy to prevent catastrophic forgetting.

- **Mixed Training Sets**: The training pipeline combines samples from both the primary (original) and secondary (new) datasets.
- **Multi-Dataset Monitoring**: A custom `DataRehearsalCallback` tracks validation IoU on both datasets simultaneously during training. This provides transparency into whether the model is retaining its original knowledge while adapting to new visual distributions.

## 8. Hardware-Aware Edge Analysis

### On-Device Profiling
Beyond software simulation, the project integrates direct hardware analysis via `scripts/stack_analyzer.py` on the ESP32-S3.

- **Stack Peak Usage**: Measures the absolute maximum memory consumed by the TFLite Micro interpreter to ensure it stays within the ~300 KB SRAM ceiling.
- **Energy per Inference**: Calculates energy consumption in milliJoules (mJ) by measuring current draw during the execution of a frame.
- **Inference Latency**: Tracks the end-to-end processing time, allowing for real-time performance validation.

### Neural Architecture Search (NAS) Monitoring
During the training phase, the architecture hooks NAS monitoring scripts into critical operational layers. This tooling permits the active tracking of latency bounds and memory overheads in Python, ensuring that all structural topologies will execute efficiently on the Rust inference engine long before the firmware is compiled.

### Rust-Powered Inference
The inference backend is implemented in Rust using a `no_std` environment. This choice provides deterministic memory management and allows for extreme optimizations like executing critical hot-loops directly from IRAM.

## 9. Optimized Spatial Upsampling

`Nano_U` has transitioned from Bilinear Interpolation to **Nearest-Neighbor (NN) Interpolation** for all upsampling operations in the decoder.

- **Arithmetic Efficiency**: Unlike Bilinear interpolation, which requires multiple multiplications and additions per pixel, NN interpolation simply duplicates existing values via memory shifts.
- **Hardware Synergy**: This optimization is natively supported by the INT8 Rust inference engine, further reducing CPU cycle count and battery drain on the ESP32.

## 10. Research Limitations & Future Directions

### Current Limitations
1. **Binary Constraint**: The current architecture is hyper-optimized for binary (background vs. crop) segmentation. Extending to multi-class requires a wider bottleneck and potential skip-connection re-introduction, which could break the ~300 KB SRAM ceiling.
2. **Fixed Resolution**: The compound scaling is mathematically tied to the $60 \times 80$ input. Scaling to $120 \times 160$ or higher would require a deeper downsampling stack or aggressive pointwise striding.

### Future Work
- **Temporal Consistency**: Integrating 1D-Temporal depthwise convolutions for video-rate segmentation stability.
- **Auto-Encoder Pruning**: Investigating SVD-based automated channel pruning during QAT to further reduce the 2.9K parameter count.
- **HIM (Hardware-In-the-Loop) NAS**: Automating the evolutionary search by feeding actual energy metrics from `stack_analyzer.py` back into the fitness function.
