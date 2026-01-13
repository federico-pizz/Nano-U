# Nano-U

Tiny segmentation inference on ESP32-S3 with Rust runtime and Python tooling.

Overview
--------
This repository combines a Rust runtime for ESP32-S3 and Python tools to build, quantize, and analyze tiny segmentation models.

Contents
--------
- `esp_flash/` — Rust project (ESP32-S3) for running inference and performing stack usage analysis.
- `models/` — TFLite models used by the Python tooling (moved here from `esp_flash/models`). Keep large models out of git if needed.
- Python scripts: `build_net.py`, `gen_image.py`, `stack_analyzer.py`, and utilities in `src/`.

Quickstart
----------
1. Generate a small INT8 TFLite model (host):

```bash
python3 build_net.py
# model saved to models/dummy5.tflite
```

2. Build and flash the Rust firmware (ESP32-S3):

```bash
cd esp_flash
cargo build --release --bin analysis
# Then from repository root
./esp_flash/run_analyzer.sh
```

3. Analyze stack usage (host):

```bash
python3 stack_analyzer.py
# Generates stack_usage.png from stack_log.txt
```

Repository Layout
-----------------
- `esp_flash/` — Rust project with `Cargo.toml`, `src/`, and build scripts.
- `models/` — Model files (.tflite) used by the examples and analyzer.
- `build_net.py` — Small TF model generator (INT8 TFLite).
- `stack_analyzer.py` — Parses UART logs and creates visualizations.

Notes & Recommendations
-----------------------
- Consider storing large models or build artifacts in releases or external storage rather than in git history.
- Keep the `rust-toolchain.toml` to ensure the correct Rust toolchain is used for cross-compilation.

License
-------
See `LICENSE` in the repo root.
