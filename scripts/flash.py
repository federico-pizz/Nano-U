"""Build and flash the firmware, with the quant-params dependency taken care of.

`firmware/build.rs` reads `<model>_quant_params.json` at compile time to bake the
INT8 scales, zero-points and normalization constants into the binary. That file
is derived from the tracked `.tflite`, so it is not committed - which means a
fresh clone cannot `cargo build` until something regenerates it. This script is
that something, which is why it is the supported way to build rather than a
convenience wrapper.

It also derives `MODELS_DIR`, `TEST_IMG_DIR` and `MODEL_NAME` from the preset, so
the three environment variables stop being something to remember.

Usage:
    python scripts/flash.py inference
    python scripts/flash.py online --preset botanicgarden
    python scripts/flash.py single_inference --target-img 42
    python scripts/flash.py capture --no-monitor      # leaves the serial port free
    python scripts/flash.py run --build-only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from nanou.config import get_project_root, resolve
from nanou.preflight import (
    PreflightError,
    quant_params_are_stale,
    quant_params_path,
    require_esp_toolchain,
    require_espflash,
    require_serial_port,
    require_tflite_model,
)

BINARIES = ["run", "inference", "single_inference", "online", "analysis",
            "analysis_person_detect", "capture"]


def regenerate_quant_params(tflite_path, models_dir, model_name, config_path):
    """Rebuild the JSON that build.rs reads, from the .tflite it describes."""
    from src.quantize_model import extract_quant_params

    params = extract_quant_params(str(tflite_path), config_path=str(config_path))
    out_path = quant_params_path(models_dir, model_name)
    out_path.write_text(json.dumps(params, indent=2))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Build and flash a firmware binary for a preset's model")
    parser.add_argument("binary", choices=BINARIES, help="firmware binary to build")
    parser.add_argument("--preset", default="tinyagri",
                        help="config preset from config/presets/ (default: tinyagri)")
    parser.add_argument("--model", default="nanou",
                        help="model block in the preset (default: nanou)")
    parser.add_argument("--build-only", action="store_true",
                        help="compile without flashing the board")
    parser.add_argument("--no-monitor", action="store_true",
                        help="flash without attaching the serial monitor, which "
                             "leaves the port free for a host-side reader")
    parser.add_argument("--target-img", type=int, default=None,
                        help="image index for the single_inference binary")
    parser.add_argument("--port", default=None,
                        help="serial port to check before flashing, e.g. /dev/ttyUSB0")
    args = parser.parse_args()

    root = get_project_root()
    resolved = resolve(args.preset)
    cfg = resolved.for_model(args.model)

    try:
        require_esp_toolchain()
        if not args.build_only:
            require_espflash()
            require_serial_port(args.port)

        tflite = require_tflite_model(cfg.models_dir, cfg.model_name, preset=args.preset)

        if quant_params_are_stale(cfg.models_dir, cfg.model_name):
            print(f"Regenerating quant params from {tflite}")
            written = regenerate_quant_params(tflite, cfg.models_dir, cfg.model_name,
                                              resolved.path)
            print(f"  wrote {written}")
        else:
            print(f"Quant params current: {quant_params_path(cfg.models_dir, cfg.model_name)}")

    except PreflightError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

    env = dict(os.environ)
    env["MODELS_DIR"] = str(cfg.models_dir)
    env["MODEL_NAME"] = cfg.model_name
    env["TEST_IMG_DIR"] = str(Path(cfg.data_root) / "test" / "img")
    if args.target_img is not None:
        env["TARGET_IMG_IDX"] = str(args.target_img)

    action = "build" if args.build_only else "run"
    cmd = ["cargo", action, "--release", "--bin", args.binary]
    if not args.build_only and args.no_monitor:
        # The cargo runner is `espflash flash --monitor`; overriding it here is
        # what keeps the port free for a host-side reader such as device_eval.
        env["CARGO_TARGET_XTENSA_ESP32S3_NONE_ELF_RUNNER"] = "espflash flash"

    print(f"\n{' '.join(cmd)}")
    print(f"  MODELS_DIR={env['MODELS_DIR']}")
    print(f"  MODEL_NAME={env['MODEL_NAME']}")
    print(f"  TEST_IMG_DIR={env['TEST_IMG_DIR']}\n")

    # cargo writes straight to the terminal, so flush first or the build output
    # appears above the lines describing the build.
    sys.stdout.flush()
    result = subprocess.run(cmd, cwd=root / "firmware", env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
