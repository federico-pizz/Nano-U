"""Precondition checks, each raising an error that names the command that fixes it.

One small predicate per prerequisite, called by the script that needs it at the
point where it would otherwise fail obscurely. There is deliberately no `doctor`
script: a separate diagnostic would be a second implementation of every check,
and it only helps people who already know it exists.

The discipline here is the whole point. Every error a user can hit must end with
the command that resolves it - if a check is added that only says what is wrong,
this module has stopped earning its place.
"""

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from .config import SPLITS

# Image extension the pipeline reads. Kept here so the dataset check and the
# error message cannot disagree about what counts as a frame.
IMAGE_SUFFIX = ".png"


class PreflightError(RuntimeError):
    """A prerequisite is missing. The message ends with the command that fixes it."""


def _fix(message: str, command: str) -> PreflightError:
    return PreflightError(f"{message}\n\n  Fix it with:  {command}\n")


def _preset_flag(preset: str | None) -> str:
    return f" --preset {preset}" if preset else ""


# -- dataset --------------------------------------------------------------------


def require_dataset(data_root, preset: str | None = None, splits: Iterable[str] = SPLITS) -> None:
    """Check that a dataset is present and laid out as the pipeline expects.

    Verifies `<data_root>/<split>/{img,mask}` for each split, that they hold
    PNGs, and that each split's image and mask counts match. A missing val split
    is tolerated - training runs without one - but a half-present one is not.

    Raises:
        PreflightError: naming the get_data command that would fix it.
    """
    root = Path(data_root)
    get_data = f"python scripts/get_data.py{_preset_flag(preset) or ' --preset <name>'}"

    if not root.is_dir():
        raise _fix(f"Dataset directory not found: {root}", get_data)

    for split in splits:
        img_dir, mask_dir = root / split / "img", root / split / "mask"
        missing = [d for d in (img_dir, mask_dir) if not d.is_dir()]
        if missing:
            if split == "val" and not (img_dir.exists() or mask_dir.exists()):
                continue  # training tolerates a dataset with no validation split
            listed = ", ".join(str(d) for d in missing)
            raise _fix(f"Dataset at {root} is missing: {listed}", get_data)

        n_img = len(list(img_dir.glob(f"*{IMAGE_SUFFIX}")))
        n_mask = len(list(mask_dir.glob(f"*{IMAGE_SUFFIX}")))
        if n_img == 0:
            raise _fix(f"No {IMAGE_SUFFIX} files in {img_dir}", get_data)
        if n_img != n_mask:
            raise _fix(
                f"{root.name}/{split} has {n_img} images but {n_mask} masks; "
                f"the pipeline pairs them by basename and needs one of each",
                get_data,
            )


# -- models ---------------------------------------------------------------------


def require_keras_model(models_dir, model_name: str, preset: str | None = None) -> Path:
    """Return the trained Keras checkpoint for *model_name*, or raise.

    Raises:
        PreflightError: naming the training command that would produce it.
    """
    directory = Path(models_dir)
    for suffix in (".h5", ".keras"):
        candidate = directory / f"{model_name}{suffix}"
        if candidate.exists():
            return candidate
    raise _fix(
        f"No trained model '{model_name}' in {directory}",
        f"python scripts/train.py{_preset_flag(preset)}",
    )


def require_tflite_model(models_dir, model_name: str, preset: str | None = None) -> Path:
    """Return the INT8 TFLite file for *model_name*, or raise.

    Raises:
        PreflightError: naming the export command that would produce it.
    """
    path = Path(models_dir) / f"{model_name}.tflite"
    if not path.exists():
        raise _fix(
            f"No INT8 TFLite model at {path}",
            f"python scripts/export.py{_preset_flag(preset)}",
        )
    return path


def require_model(models_dir, model_name: str, preset: str | None = None) -> Path:
    """Return whichever model file exists for *model_name*, preferring the INT8 one.

    Evaluation runs either format through the same forward path, so it only
    needs one of them present.

    Raises:
        PreflightError: naming the training command that would produce it.
    """
    directory = Path(models_dir)
    for suffix in (".tflite", ".h5", ".keras"):
        candidate = directory / f"{model_name}{suffix}"
        if candidate.exists():
            return candidate
    raise _fix(
        f"No model '{model_name}' in {directory} (looked for .tflite, .h5, .keras)",
        f"python scripts/train.py{_preset_flag(preset)}",
    )


def require_teacher(teacher_weights, preset: str | None = None) -> Path:
    """Return the teacher checkpoint a distilling student reads, or raise.

    Needed wherever a student distils from a teacher it did not just train.
    Training itself only prints a line and goes on with a randomly initialised
    teacher, which is a distillation run in name only.

    Raises:
        PreflightError: naming the command that trains the teacher.
    """
    if teacher_weights and Path(teacher_weights).exists():
        return Path(teacher_weights)
    raise _fix(
        f"No teacher checkpoint at {teacher_weights}"
        if teacher_weights
        else "The preset sets no teacher_weights for the student",
        f"python scripts/train.py --no-distill --model bu_net{_preset_flag(preset)}",
    )


def require_free_outputs(paths: Iterable, rerun: str) -> None:
    """Refuse to start a run that would replace trained models already on disk.

    A shipped preset trains into the directory the tracked models live in, so a
    run started to try something out would silently replace them.

    Raises:
        PreflightError: listing the files, naming the rerun with --overwrite.
    """
    existing = [Path(p) for p in paths if Path(p).exists()]
    if existing:
        listed = "\n".join(f"    {p}" for p in existing)
        raise _fix(
            f"This run would overwrite:\n{listed}\n\n"
            "To keep them, copy the preset and point models_dir somewhere else",
            f"{rerun} --overwrite",
        )


def quant_params_path(models_dir, model_name: str) -> Path:
    """Where `firmware/build.rs` looks for the calibration parameters."""
    return Path(models_dir) / f"{model_name}_quant_params.json"


def quant_params_are_stale(models_dir, model_name: str) -> bool:
    """True when the quant-params JSON is missing or older than its .tflite.

    Not a raising check: it is a **derived** file, so the right response is to
    regenerate it rather than to make the user do it. `scripts/flash.py` calls
    this and rebuilds; nothing else needs to care.
    """
    params = quant_params_path(models_dir, model_name)
    if not params.exists():
        return True
    tflite = Path(models_dir) / f"{model_name}.tflite"
    return tflite.exists() and params.stat().st_mtime < tflite.stat().st_mtime


# -- firmware toolchain ---------------------------------------------------------

ESP_ENV_SCRIPT = "source ~/export-esp.sh"


def require_esp_toolchain() -> None:
    """Check that the Xtensa toolchain is on PATH and the esp env was sourced.

    `export-esp.sh` puts `xtensa-esp-elf` on PATH and sets `LIBCLANG_PATH`; a
    shell that has not sourced it builds host binaries or fails deep inside a
    dependency's build script, neither of which points at the real cause.

    Raises:
        PreflightError: naming the script to source.
    """
    if shutil.which("cargo") is None:
        raise _fix(
            "cargo is not on PATH, so the firmware cannot be built",
            "install Rust from https://rustup.rs, then " + ESP_ENV_SCRIPT,
        )
    if not os.environ.get("LIBCLANG_PATH"):
        raise _fix(
            "LIBCLANG_PATH is unset, so the ESP toolchain environment has not "
            "been sourced in this shell",
            ESP_ENV_SCRIPT,
        )


def require_espflash() -> None:
    """Check that `espflash` is installed, since cargo's runner invokes it.

    Raises:
        PreflightError: naming the install command.
    """
    if shutil.which("espflash") is None:
        raise _fix(
            "espflash is not on PATH; `cargo run` uses it as the runner for the xtensa target",
            "cargo install espflash",
        )


# -- serial ---------------------------------------------------------------------


def require_serial_port(port: str | None = None) -> None:
    """Check that the board's serial port exists and is not already held open.

    A port held by another process is the single most common device-side
    failure, and it surfaces as a timeout several seconds into a run rather than
    as a message about the monitor that is still attached.

    Raises:
        PreflightError: naming what to close.
    """
    if port is None:
        return
    device = Path(port)
    if not device.exists():
        raise _fix(
            f"Serial port {port} does not exist; the board may be unplugged or "
            f"enumerated under a different name",
            "ls /dev/ttyUSB* /dev/ttyACM*",
        )
    try:
        fd = os.open(port, os.O_RDWR | os.O_NONBLOCK)
    except OSError as exc:
        raise _fix(
            f"Cannot open serial port {port}: {exc}. Another process is "
            f"probably holding it - a `--monitor` session from a previous flash "
            f"is the usual culprit",
            f"fuser -k {port}",
        ) from None
    os.close(fd)


__all__ = [
    "PreflightError",
    "require_dataset",
    "require_keras_model",
    "require_model",
    "require_tflite_model",
    "require_free_outputs",
    "quant_params_path",
    "quant_params_are_stale",
    "require_esp_toolchain",
    "require_espflash",
    "require_serial_port",
]
