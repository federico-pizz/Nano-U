"""On-device evaluation: flash the eval binary, read the serial output, score it."""

import argparse
import sys
from pathlib import Path

from nanou.config import load_config, preset_path
from nanou.device.evaluate import evaluate_esp_outputs, run_inference_on_device
from nanou.preflight import (
    PreflightError,
    require_esp_toolchain,
    require_tflite_model,
)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Nano-U model on ESP32 hardware")
    parser.add_argument("model", default="nanou", nargs="?", help="Model to benchmark")
    parser.add_argument(
        "--preset",
        default="tinyagri",
        help="config preset from config/presets/ (default: tinyagri)",
    )
    parser.add_argument(
        "--config", default=None, help="explicit config file, used instead of --preset"
    )
    args = parser.parse_args()
    config_path = args.config or str(preset_path(args.preset))
    try:
        require_esp_toolchain()
        require_tflite_model(
            load_config(config_path)["data"]["paths"]["models_dir"], args.model, preset=args.preset
        )
    except PreflightError as e:
        raise SystemExit(f"\n{e}") from e

    repo_root = Path(__file__).parent.parent.resolve()
    print("==========================================")
    print(f"ESP32 Benchmarking Pipeline ({args.model})")
    print("==========================================")

    device_result = run_inference_on_device(
        repo_root, model_name=args.model, config_path=config_path
    )
    if not device_result or not device_result.get("img_data"):
        print("Failed to collect inference data from ESP32.")
        sys.exit(1)

    print("\n[3/3] Evaluating ESP32 outputs and calculating metrics...")
    evaluate_esp_outputs(device_result, repo_root, model_name=args.model, config_path=config_path)


if __name__ == "__main__":
    main()
