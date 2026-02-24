# Nano-U: Comprehensive Features and Architecture Guide

This document provides a deep dive into the theoretical and practical underpinnings of the Nano-U repository. It details the neural network layers, the Neural Architecture Search (NAS) strategy, Knowledge Distillation, and the core scripts that drive the pipeline. Where applicable, the foundational academic papers that introduced these concepts are cited.

---

## 1. Core Model Architectures

The project revolves around two primary architectures: the heavy **Teacher (BU-Net)** and the ultra-lightweight **Student (Nano-U)**.

### 1.1 Nano-U (Student)
Nano-U is designed for extreme parameter efficiency (typically < 3,000 parameters) to run on embedded devices like the ESP32-S3. It is an encoder-decoder network that intentionally drops skip-connections to minimize memory overhead during inference.
- **Reference Pattern:** Classic Autoencoder / Fully Convolutional Network (FCN).
- *Paper:* [Fully Convolutional Networks for Semantic Segmentation (Long et al., 2015)](https://arxiv.org/abs/1411.4038)

### 1.2 BU-Net (Teacher)
BU-Net serves as the high-capacity teacher model. It uses a traditional U-Net symmetrical topology (encoder-decoder with skip connections) to generate high-quality soft targets for the student to mimic.
- **Reference Pattern:** U-Net.
- *Paper:* [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)

---

## 2. Neural Network Building Blocks (Primitives)

To drastically reduce the parameter count, Nano-U abandons standard 2D convolutions in favor of parameter-efficient primitives. These primitives serve as the "search space" for the NAS engine.

### 2.1 Depthwise Separable Convolution (`depthwise_sep_conv`)
Splits a standard convolution into two distinct steps: a Depthwise convolution (filtering each channel separately) followed by a Pointwise (1x1) convolution (combining channels). This drastically reduces both parameters and Multiply-Accumulate operations (MACs).
- **Function:** `src/models/layers.py::depthwise_sep_conv`
- *Paper:* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (Howard et al., 2017)](https://arxiv.org/abs/1704.04861)

### 2.2 Bottleneck Depthwise Convolution (`bottleneck_dw_conv`)
Also known as an **Inverted Residual Block**. It first expands the channel dimension using a 1x1 pointwise conv, applies a 3x3 depthwise conv in the higher-dimensional space, and then projects it back down to a smaller channel dimension using another 1x1 conv, adding a residual skip connection.
- **Function:** `src/models/layers.py::bottleneck_dw_conv`
- *Paper:* [MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al., 2018)](https://arxiv.org/abs/1801.04381)

### 2.3 Residual Depthwise Convolution (`residual_dw_conv`)
A custom block built for this repository. It applies a Pointwise → Depthwise → Pointwise sequence, injecting Batch Normalization after every step and closing with a residual skip connection. It provides a highly accurate, parameter-efficient middle-ground without expanding channels like MobileNetV2.
- **Function:** `src/models/layers.py::residual_dw_conv`
- *Concept:* Combines standard residual learning ([ResNet, He et al., 2015](https://arxiv.org/abs/1512.03385)) with MobileNet logic.

### 2.4 Triple Convolution (`triple_conv`)
The most expressive (and parameter-heavy) primitive in the Nano-U search space. It sequentially chains three MobileNet-style Depthwise-Separable blocks to maximize receptive field and feature extraction capability at a given stage.
- **Function:** `src/models/layers.py::triple_conv`

---

## 3. Evolutionary Neural Architecture Search (NAS)

Instead of manually designing the layer sequence, the repository uses an **Evolutionary Algorithm** to search for the best combination of the 4 primitives across the network's 7 conceptual stages (3 encoder blocks, 1 bottleneck, 3 decoder blocks).

### 3.1 Evolutionary Search Loop
The algorithm initializes a population of random architectures (e.g., `[0, 2, 3, 1, 0, 0, 0]`), proxies a fast training loop, scores their fitness, and uses mutation to evolve the population over generations.
- **Class:** `src/nas.py::NASSearcher`
- *Paper:* [Regularized Evolution for Image Classifier Architecture Search (AmoebaNet) (Real et al., 2019)](https://arxiv.org/abs/1802.01548)

### 3.2 SVD-Based Redundancy Metric
The fitness function doesn't just evaluate `val_iou`. It actively penalizes "redundant" architectures. During proxy training, it hooks into the intermediate layer activations and performs **Singular Value Decomposition (SVD)**. If a small number of singular values explain the vast majority of the variance (high condition number, low rank), the layer is deemed redundant and penalized by the fitness function.
- **Functions:** `src/nas.py::calculate_redundancy`, `score_architecture`
- *Paper:* Inspired by rank-based filter pruning techniques, e.g., [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration (He et al., 2019)](https://arxiv.org/abs/1811.00250) and [Network Trimming (Hu et al., 2016)](https://arxiv.org/abs/1607.03250).

---

## 4. Knowledge Distillation

To force the tiny student model to behave like a much larger, more capable model, the training pipeline utilizes Knowledge Distillation. The loss function compares the Student's outputs not only to the hard Ground Truth masks, but also to the "soft" probability logits generated by the frozen Teacher model.
- **Function:** `src/train.py::train_with_distillation`
- *Paper:* [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)

---

## 5. Post-Training Quantization (PTQ)

To deploy Nano-U to microcontrollers lacking floating-point units or vast memory spaces, the model must be converted to an **INT8** representation.
- **Calibration:** `src/quantize_model.py` loads the actual processed validation dataset and passes it to the `TFLiteConverter`. This "Representative Dataset" allows the converter to track activation statistics (min/max ranges) so that the fp32 values can be mapped accurately into the 8-bit integer space [-128, 127] without severe accuracy degradation.
- **Function:** `src/models/utils.py::convert_to_tflite_quantized`

---

## 6. Core Modules and Main Functions Under the Hood

To fully understand how the repository executes these concepts, let's look at the primary functions handling the workload.

### 6.1 Training Orchestration (`src/train.py`)

- **`train_single_model(model, config, train_data, val_data)`**
  This is the standard Keras `.fit()` wrapper. It handles:
  - Extracting hyperparameter configurations (epochs, batch size, learning rate).
  - Injecting the `NASCallback` to monitor SVD redundancy over time if enabled in the config.
  - Setting up standard callbacks (ModelCheckpoint for `val_iou`, EarlyStopping, ReduceLROnPlateau).
  - **Logic:** Compiles the model with Binary Crossentropy and the custom `BinaryIoU` metric, then trains it on the provided `tf.data.Dataset`.

- **`train_with_distillation(student, teacher, config, train_data, val_data)`**
  This function implements the distillation logic.
  - **Logic:** It defines a custom training step (`train_step` inner function) that overrides the default `.fit()` loop using `tf.GradientTape()`. 
  - For each batch, it computes the Teacher's predictions (soft targets) and the Student's predictions.
  - The final loss is a weighted sum (controlled by `alpha`) of the Student-vs-Teacher loss (Mean Squared Error) and the Student-vs-GroundTruth loss (Binary Crossentropy).
  - The temperature parameter softens the logits before MSE calculation to make transferring dark knowledge easier.

### 6.2 Neural Architecture Search (`src/nas.py`)

- **`NASSearcher.search(self, train_fn)`**
  The main engine for Evolutionary architecture search.
  - **Logic:** Maintains a `population` of arrays (e.g., `[0, 2, 3, 1, 0, 0, 0]`) where each integer maps to a specific block (like `depthwise_sep_conv` or `residual_dw_conv`).
  - At each generation, it asks the user-supplied `train_fn` (a proxy standard or distillation training script running for minimal epochs) to evaluate every architecture in the population.
  - It assigns a fitness score based on `val_iou` and parameter efficiency, selects the top contenders, mutates them (randomly flipping integers in the array) to form the next generation.

- **`compute_layer_redundancy(activations, eps)`**
  The mathematical core of the SVD redundancy penalty.
  - **Logic:** Flattens the layer activations obtained from a specific feature map. Calculates the covariance matrix and computes its Singular Values using `tf.linalg.svd`. 
  - It tracks the **Condition Number** (ratio of largest to smallest singular value) and **Effective Rank** (how many singular values make up 99% of the variance). 
  - If the rank is low and the condition number is high, the layer is doing redundant work (e.g., repeating the same feature detectors) and gets flagged.

### 6.3 Pipeline Execution (`src/pipeline.py`)

- **`run_training_pipeline(config_name, config_path, output_dir)`**
  The high-level entry point used by `scripts/train_standard.py`.
  - **Logic:** Looks up the requested experiment name in `config/experiments.yaml`, builds the output directories, invokes `src/train.py::train_model`, and then neatly saves the execution history and the resulting `.keras` file to the specified `results/` folder.

- **`run_nas_search(config_path, output_dir, model_name)`**
  The high-level entry point for architecture search.
  - **Logic:** Instantiates the `NASSearcher`, builds a custom `train_proxy` function that trains candidates for a very short duration (e.g., 2-4 epochs), and passes this proxy to the searcher. It tracks all generations and outputs the `best_arch.json` file.

- **`quantize_and_benchmark(keras_path, models_dir)`**
  The post-training optimization script.
  - **Logic:** Takes a `.keras` model, calls `quantize_model.py` (which runs inference on real validation data to calibrate the INT8 scales), saves a `.tflite` file, and finally runs `src/benchmarks.py` to calculate the throughput (FPS) and latency (ms) of the quantized model on the host CPU.
