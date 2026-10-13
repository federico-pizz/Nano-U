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
import sys

from nanou.device.flash import BINARIES, build
from nanou.preflight import PreflightError


def main():
    parser = argparse.ArgumentParser(
        description="Build and flash a firmware binary for a preset's model"
    )
    parser.add_argument("binary", choices=BINARIES, help="firmware binary to build")
    parser.add_argument(
        "--preset",
        default="tinyagri",
        help="config preset from config/presets/ (default: tinyagri)",
    )
    parser.add_argument(
        "--model", default="nanou", help="model block in the preset (default: nanou)"
    )
    parser.add_argument(
        "--build-only", action="store_true", help="compile without flashing the board"
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="flash without attaching the serial monitor, which "
        "leaves the port free for a host-side reader",
    )
    parser.add_argument(
        "--target-img", type=int, default=None, help="image index for the single_inference binary"
    )
    parser.add_argument(
        "--port", default=None, help="serial port to check before flashing, e.g. /dev/ttyUSB0"
    )
    args = parser.parse_args()

    try:
        code = build(
            args.binary,
            preset=args.preset,
            model=args.model,
            build_only=args.build_only,
            no_monitor=args.no_monitor,
            target_img=args.target_img,
            port=args.port,
        )
    except PreflightError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(code)


if __name__ == "__main__":
    main()
