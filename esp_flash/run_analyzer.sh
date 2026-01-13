#!/bin/bash
# Complete stack analysis workflow using On-Device Stack Painting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Stack Usage Analysis Workflow"
echo "Repository root: $REPO_ROOT"
echo "=========================================="
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
echo "[2/4] Flashing and running on device..."
LOG="$REPO_ROOT/stack_log.txt"
echo "Capturing output to $LOG..."

# Remove old log
rm -f "$LOG"

# Run in background (use espflash to avoid cargo run TTY issues)
espflash flash --monitor target/xtensa-esp32s3-none-elf/release/analysis < /dev/null > "$LOG" 2>&1 &
PID=$!

echo "Waiting for analysis to complete (PID: $PID)..."

# Wait loop
MAX_RETRIES=300 # 300 seconds timeout (5 minutes)
COUNT=0
SUCCESS=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    if grep -q "ANALYSIS_DONE" "$LOG"; then
        echo "✓ Analysis finished successfully!"
        SUCCESS=1
        break
    fi

    # Check if process died early
    if ! kill -0 $PID 2>/dev/null; then
        echo "Error: Process exited unexpectedly."
        break
    fi

    sleep 1
    ((COUNT++))
    echo -n "."
done
echo ""

# Kill the process (espflash monitor)
pkill -P $PID 2>/dev/null || true
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true

# Restore terminal settings just in case
stty sane 2>/dev/null || true

if [ $SUCCESS -eq 0 ]; then
    echo "Error: Analysis timed out or failed."
    echo "Last 50 lines of log:"
    tail -n 50 "$LOG"
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
