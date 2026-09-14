"""Build and flash a firmware binary, keeping the quant-params dependency honest.

`firmware/build.rs` reads `<model>_quant_params.json` at compile time and bakes
the INT8 scales, zero-points and normalisation constants into the binary. That
file is derived from the tracked `.tflite` and is therefore not committed, so a
fresh clone cannot `cargo build` until something regenerates it. That is what
this module is for, and why it is the supported way to build rather than a
convenience wrapper.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from nanou.config import get_project_root, resolve
from nanou.preflight import (
    quant_params_are_stale,
    quant_params_path,
    require_esp_toolchain,
    require_espflash,
    require_serial_port,
    require_tflite_model,
)
from nanou.quantize import extract_quant_params

BINARIES = [
    "run",
    "inference",
    "single_inference",
    "online",
    "analysis",
    "analysis_person_detect",
    "capture",
]


def regenerate_quant_params(tflite_path, models_dir, model_name, config_path):
    """Rebuild the JSON that build.rs reads, from the .tflite it describes."""
    params = extract_quant_params(str(tflite_path), config_path=str(config_path))
    out_path = quant_params_path(models_dir, model_name)
    out_path.write_text(json.dumps(params, indent=2))
    return out_path


def build(
    binary,
    preset="tinyagri",
    model="nanou",
    build_only=False,
    no_monitor=False,
    target_img=None,
    port=None,
):
    """Check the preconditions, refresh the quant params, then run cargo.

    Returns cargo's exit code. Raises `PreflightError` when a precondition
    fails, with a message naming the command that fixes it.
    """
    root = get_project_root()
    resolved = resolve(preset)
    cfg = resolved.for_model(model)

    require_esp_toolchain()
    if not build_only:
        require_espflash()
        require_serial_port(port)

    tflite = require_tflite_model(cfg.models_dir, cfg.model_name, preset=preset)

    if quant_params_are_stale(cfg.models_dir, cfg.model_name):
        print(f"Regenerating quant params from {tflite}")
        written = regenerate_quant_params(tflite, cfg.models_dir, cfg.model_name, resolved.path)
        print(f"  wrote {written}")
    else:
        print(f"Quant params current: {quant_params_path(cfg.models_dir, cfg.model_name)}")

    env = dict(os.environ)
    env["MODELS_DIR"] = str(cfg.models_dir)
    env["MODEL_NAME"] = cfg.model_name
    env["TEST_IMG_DIR"] = str(Path(cfg.data_root) / "test" / "img")
    if target_img is not None:
        env["TARGET_IMG_IDX"] = str(target_img)

    action = "build" if build_only else "run"
    cmd = ["cargo", action, "--release", "--bin", binary]
    if not build_only and no_monitor:
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
    return subprocess.run(cmd, cwd=root / "firmware", env=env).returncode
