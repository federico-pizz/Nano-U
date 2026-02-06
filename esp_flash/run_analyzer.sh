#!/bin/bash
# Complete stack analysis workflow using On-Device Stack Painting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Stack Usage Analysis Workflow"
echo "Repository root: $REPO_ROOT"
echo "=========================================="

# Source ESP-IDF environment if available
if [ -f "$HOME/export-esp.sh" ]; then
    echo "Sourcing ESP-IDF environment from $HOME/export-esp.sh"
    source "$HOME/export-esp.sh"
fi

echo ""

# Step 1: Build the project (run inside esp_flash)
echo "[1/4] Building project for ESP32-S3..."
cd "$SCRIPT_DIR"

cargo build --release --bin analysis
if [ $? -ne 0 ]; then
    echo "Error: Build failed!"
    exit 1
fi
echo "✓ Build complete"
echo ""

# Step 2: Run on Device and Capture Output
LOG="stack_log.txt"

# Run analysis wrapper (handles execution, logging, and termination)
echo "Starting analysis..."
python3 "$SCRIPT_DIR/run_inference.py"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Analysis failed or timed out."
    exit 1
fi

echo "✓ Capture complete"
echo ""

# Step 3: Parse Output
echo "[3/4] Parsing log..."
grep "STACK_PEAK" "$LOG" || {
    echo "Error: Could not find STACK_PEAK in log. Did the program run?"
    tail -n 200 "$LOG"
    exit 1
}

# Step 4: Run Python visualization
# Prefer using the TF venv created for host tasks (.venv-tf)
VENV_PY="$REPO_ROOT/.venv-tf/bin/python"
VENV_PIP="$REPO_ROOT/.venv-tf/bin/pip"

if [ ! -x "$VENV_PY" ]; then
    echo "Virtualenv $REPO_ROOT/.venv-tf not found or missing python. Creating minimal venv..."
    python3 -m venv "$REPO_ROOT/.venv-tf"
fi

# Install required Python packages into the venv (if missing)
if [ -x "$VENV_PIP" ]; then
    "$VENV_PIP" install --upgrade pip >/dev/null 2>&1 || true
    "$VENV_PIP" install -q matplotlib numpy || true
else
    echo "Warning: pip not found in $REPO_ROOT/.venv-tf; run: python3 -m venv $REPO_ROOT/.venv-tf and install dependencies manually"
fi

# Copy the captured log into the esp_flash folder so the analyzer finds it
cp -f "$LOG" "$SCRIPT_DIR/stack_log.txt" || true

echo "[4/4] Generating visualizations with Python..."
"$VENV_PY" "$SCRIPT_DIR/stack_analyzer.py"
if [ $? -ne 0 ]; then
    echo "Error: Python visualization failed!"
    exit 1
fi

echo ""

echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo "Generated files:"
echo "  - $LOG       (Raw UART output)"
echo "  - $SCRIPT_DIR/stack_usage.png     (Visualization)"
