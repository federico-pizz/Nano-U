#!/usr/bin/env bash
set -euo pipefail

# TF automation pipeline — assumes an existing Python venv at .venv-tf.
# Automates: image prep, training (two models), quantization and evaluation.
# Usage: ./scripts/tf_pipeline.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_DIR=".venv-tf"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_CMD=("$PYTHON_BIN" -m pip)

echo "[tf_pipeline] Working in: $ROOT"

# Require existing venv
if [ ! -d "$VENV_DIR" ]; then
  echo "[tf_pipeline] Error: virtual environment $VENV_DIR not found. Create it first:" >&2
  echo "  python3 -m venv $VENV_DIR" >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[tf_pipeline] Error: python binary not found in $PYTHON_BIN" >&2
  exit 1
fi

# Ensure pip is available and upgraded
"${PIP_CMD[@]}" install --upgrade pip || true

# Install lightweight deps only if missing
if ! "$PYTHON_BIN" -c "import numpy" >/dev/null 2>&1; then
  echo "[tf_pipeline] Installing numpy"
  "${PIP_CMD[@]}" install numpy || true
fi
if ! "$PYTHON_BIN" -c "import matplotlib" >/dev/null 2>&1; then
  echo "[tf_pipeline] Installing matplotlib"
  "${PIP_CMD[@]}" install matplotlib || true
fi

# TensorFlow is heavy — install only if missing
if ! "$PYTHON_BIN" -c "import tensorflow" >/dev/null 2>&1; then
  echo "[tf_pipeline] TensorFlow not present in venv. Installing tensorflow (may take long)"
  "${PIP_CMD[@]}" install tensorflow || true
else
  echo "[tf_pipeline] TensorFlow already available in venv"
fi

# Ensure models dir
mkdir -p models

# --- Step 1: Image prep ---
if [ -f gen_image.py ]; then
  echo "[tf_pipeline] Generating sample input image (gen_image.py)"
  "$PYTHON_BIN" gen_image.py || echo "[tf_pipeline] gen_image.py failed"
else
  echo "[tf_pipeline] gen_image.py not found — skipping image prep"
fi

# --- Step 2: Build tiny INT8 model ---
if [ -f build_net.py ]; then
  echo "[tf_pipeline] Building tiny INT8 model (build_net.py)"
  "$PYTHON_BIN" build_net.py || echo "[tf_pipeline] build_net.py failed"
else
  echo "[tf_pipeline] build_net.py not found — skipping tiny model generation"
fi

# --- Step 3: Train models ---
TRAIN_SCRIPT="src/train.py"
if [ -f "$TRAIN_SCRIPT" ]; then
  for MODEL_NAME in bu_net nano_u; do
    echo "[tf_pipeline] Training model: $MODEL_NAME"
    # Provide MODEL env var; training script should read it or fallback
    MODEL="$MODEL_NAME" "$PYTHON_BIN" "$TRAIN_SCRIPT" || echo "[tf_pipeline] Training $MODEL_NAME failed"
  done
else
  echo "[tf_pipeline] Training script $TRAIN_SCRIPT not found — skipping training"
fi

# --- Step 4: Quantization ---
QUANT_SCRIPT="src/quantize.py"
if [ -f "$QUANT_SCRIPT" ]; then
  shopt -s nullglob
  # Quantize models with .keras or .h5 extensions
  for K in models/*.keras models/*.h5; do
    [ -f "$K" ] || continue
    BASENAME=$(basename "$K")
    BASENAME="${BASENAME%.*}"
    OUT="models/${BASENAME}.tflite"
    echo "[tf_pipeline] Quantizing $K -> $OUT"
    "$PYTHON_BIN" "$QUANT_SCRIPT" --model-name "$BASENAME" --output "$OUT" --int8 --config "config/config.yaml" || echo "[tf_pipeline] Quantize failed for $K"
  done
  shopt -u nullglob
else
  echo "[tf_pipeline] Quantize script not found — skipping quantization"
fi

# --- Step 5: Evaluation ---
EVAL_SCRIPT="src/evaluate.py"
if [ -f "$EVAL_SCRIPT" ]; then
  shopt -s nullglob
  # Create results/eval folder
  mkdir -p results/eval
  for TFL in models/*.tflite; do
    [ -f "$TFL" ] || continue
    BASENAME=$(basename "$TFL")
    BASENAME="${BASENAME%.*}"
    OUT_PNG="results/eval/${BASENAME}_eval.png"
    echo "[tf_pipeline] Evaluating model: $TFL -> $OUT_PNG"
    "$PYTHON_BIN" "$EVAL_SCRIPT" --model-name "$BASENAME" --out "$OUT_PNG" || echo "[tf_pipeline] Evaluation failed for $TFL"
  done
  shopt -u nullglob
else
  echo "[tf_pipeline] Evaluation script $EVAL_SCRIPT not found — skipping evaluation"
fi

# Final listing
echo "[tf_pipeline] Models available:"
ls -la models || true

echo "[tf_pipeline] Done. Activate venv with: source $VENV_DIR/bin/activate"
