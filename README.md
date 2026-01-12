# Nano-U: Efficient Semantic Segmentation for Robotic Navigation in Outdoor Environments

## Overview

This repository contains the TensorFlow re-implementation and extensions of the Nano-U model (thesis project). The goal is on-device binary semantic segmentation (traversable ground) for resource-constrained platforms such as the ESP32.

Key points:
- Lightweight segmentation model optimized for microcontrollers.
- Full TensorFlow training, quantization and inference pipeline in `src/`.
- Legacy artifacts and a previous MicroFlow/Rust implementation are preserved in `old/` for reference.
- A Rust-based conversion step (TFLite -> MicroFlow/Rust) is expected to live under `rust/convert_tflite_to_microflow/` (placeholder).

## Quick links
- Config: [`config/config.yaml`](config/config.yaml:1)
- Training script: [`src/train_tf.py`](src/train_tf.py:1)
- Quantization / TFLite export: [`src/quantize_tf.py`](src/quantize_tf.py:1)
- Inference (TF): [`src/infer_tf.py`](src/infer_tf.py:1)
- Data preparation: [`src/prepare_data.py`](src/prepare_data.py:1)
- Data utilities: [`src/utils/data_tf.py`](src/utils/data_tf.py:1)
- Old project (PyTorch / MicroFlow examples): [`old/`](old/:1)

## Project structure (high-level)
- [`config/config.yaml`](config/config.yaml:1) — Pipeline and model configuration (input shape, dataset paths, training and quantization settings).
- [`src/`] — TensorFlow implementation: training, quantization, inference, data preparation and utilities.
- [`models/`] — (created at runtime) trained Keras models and converted TFLite artifacts.
- [`notebooks/`] — experimental notebooks (retraining experiments).
- [`old/`] — previous project code and MicroFlow examples kept for reference.
- [`rust/convert_tflite_to_microflow/`] — placeholder for the Rust converter (to be added by user).

## Quickstart

1. Create a Python virtual environment and install dependencies (recommended `.venv-tf` for TensorFlow):

```bash
python -m venv .venv-tf
source .venv-tf/bin/activate
pip install -r [`requirements.txt`](requirements.txt:1)
```

2. Configure dataset paths in [`config/config.yaml`](config/config.yaml:1).

3. Prepare data (resizes, sequential splits):

```bash
python [`src/prepare_data.py`](src/prepare_data.py:1) --config config/config.yaml
```

4. Train model (example for Nano_U). The project supports running the training script directly or as a module; prefer `python -m src.train_tf` when using the package context:

```bash
# Direct run (works thanks to import guard):
python src/train_tf.py --config config/config.yaml --model nano_u

# Or module run (recommended):
python -m src.train_tf --config config/config.yaml --model nano_u
```

For knowledge distillation (train student with a pretrained teacher):

```bash
python [`src/train_tf.py`](src/train_tf.py:1) --config config/config.yaml --model nano_u --distill --teacher-weights models/bu_net_tf_best.keras
```

5. Convert trained Keras model to TFLite (optional INT8 quantization):

```bash
python [`src/quantize_tf.py`](src/quantize_tf.py:1) --config config/config.yaml --model-path models/nano_u_tf_best.keras --int8
```

6. Run TensorFlow inference on validation set:

```bash
python [`src/infer_tf.py`](src/infer_tf.py:1) --config config/config.yaml --weights models/nano_u_tf_best.keras
```

The scripts are intentionally CLI-friendly and read settings from [`config/config.yaml`](config/config.yaml:1). Override parameters via CLI flags when necessary.

## Quantization and TFLite

- INT8 conversion uses a representative dataset (validation split) for calibration. Settings live under the `quantization` section of [`config/config.yaml`](config/config.yaml:1).
- Output TFLite files are written to `models/` by default. Use [`src/quantize_tf.py`](src/quantize_tf.py:1) to customize input/output types and supported ops.

## Rust / MicroFlow conversion (placeholder)

A conversion step from TFLite to a Rust/MicroFlow inference artifact is expected. This repository currently assumes that a folder such as [`rust/convert_tflite_to_microflow/README.md`](rust/convert_tflite_to_microflow/README.md:1) will be added later with details.

Expected inputs for that converter:
- TFLite model file (e.g., `models/nano_u_int8.tflite`)
- Optional metadata (input shape, mean/std normalization)

Expected outputs:
- Rust crate or library wrapping MicroFlow inference kernels
- Static binary or firmware artifacts for flashing to ESP32

NOTE: Paste your Rust conversion folder under `rust/convert_tflite_to_microflow/` when available; this README contains placeholders and will be linked from here.

## Deploying to ESP32 / Microcontrollers

This project targets small devices; previously working examples using MicroFlow are kept in [`old/MicroFlow_implementation/`](old/MicroFlow_implementation/:1) and an ESP flash example in [`old/esp-flash/`](old/esp-flash/:1). When the Rust converter is added it should document required toolchains (cargo, xtensa toolchain or espidf) and flashing steps.

## Reproducibility notes

- The pipeline reads paths and hyperparameters from [`config/config.yaml`](config/config.yaml:1). Seed is set in training config.
- Data normalization: configured in [`config/config.yaml`](config/config.yaml:1) and applied consistently by [`src/utils/data_tf.py`](src/utils/data_tf.py:1).
- If you need to reproduce results exactly, export the `models/` directory after training and include the exact `config/config.yaml` used.

## Legacy / Reference

- Old PyTorch training and helper scripts are under [`old/`](old/:1) for reference during migration. In particular see [`old/pytorch-tflite-quant.py`](old/pytorch-tflite-quant.py:1) and [`old/model_training.py`](old/model_training.py:1).

## Contribution

Contributions are welcome. Please follow the standard GitHub workflow (fork, feature branch, PR). Keep changes to configuration and model artifacts well documented.

## License

This project is MIT licensed — see [`LICENSE`](LICENSE:1).

---

Last updated: concise README with TensorFlow pipeline and placeholder for Rust conversion. Please paste the Rust conversion folder under `rust/convert_tflite_to_microflow/` and I will update the README to include exact conversion and flashing steps.
