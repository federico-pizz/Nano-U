"""The pipeline: train -> export INT8 -> evaluate, and the two steps it is built from.

`run_pipeline` is what `scripts/train.py` runs. `run_training` (one model,
trained and its `results.json` written) is also what the CV sweep in
`nanou.search` calls per fold, and `export_int8` is the export step on its own.

Nothing here catches: a failure anywhere arrives as the exception that caused
it, with its traceback.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from nanou.metrics import NumpyEncoder
from nanou.train import train_model

# The teacher never runs on the MCU, so it is never exported or scored as INT8.
TEACHER = "bu_net"
STUDENT = "nanou"

# Written by every run_pipeline call, under the run's output directory.
SUMMARY_FILE = "summary.json"


def _select_best_epoch_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Collapse a per-epoch history dict to a single-value summary.

    Val metrics are taken at the epoch with the lowest val_loss (the saved
    checkpoint epoch). Train metrics are taken at the final epoch.
    """
    val_losses = raw.get("val_loss")
    best_epoch = int(np.argmin(val_losses)) if isinstance(val_losses, list) and val_losses else None
    out = {}
    for k, v in raw.items():
        if isinstance(v, list) and len(v) > 0:
            out[k] = v[best_epoch] if best_epoch is not None and k.startswith("val_") else v[-1]
        else:
            out[k] = v
    return out


def export_int8(
    keras_path: str | Path,
    models_dir: str | Path = "models/",
    config_path: str = "config/presets/tinyagri.yaml",
    calibration: str | None = None,
) -> dict[str, Any]:
    """Convert a Keras model to a calibrated INT8 TFLite file.

    Also produces a companion _quant_params.json consumed by firmware/build.rs
    at compile time to embed correct scale/zero-point values in the binary.

    config_path must be the SAME config the model was trained with: it supplies
    the calibration image directory, the input shape and the normalization
    mean/std. Pointing it at a different config silently calibrates on the wrong
    (or missing) data, which quietly wrecks the INT8 scales.

    calibration overrides the configured calibration source for this export:
    "images" or "noise". The calibration ablation exports the same model both
    ways; nothing else should ask for noise.

    Returns:
        ``tflite_path``, ``quant_params_path`` and ``size_kb``.
    """
    from nanou.quantize import quantize_model

    keras_path = Path(keras_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_dest = models_dir / keras_path.name
    tflite_path = models_dir / f"{keras_path.stem}.tflite"

    if keras_path != model_dest:
        shutil.copy(keras_path, model_dest)
        print(f"Copied {keras_path.name} -> {model_dest}")

    print(f"Quantizing {keras_path.stem} -> INT8 TFLite...")
    quant_params_path = quantize_model(
        str(model_dest), str(tflite_path), config_path=config_path, calibration=calibration
    )
    size_kb = round(tflite_path.stat().st_size / 1024, 2)
    print(f"TFLite export complete: {tflite_path} ({size_kb} KB)")
    print(f"Quant params saved:     {quant_params_path}")
    print("     firmware/build.rs reads this file at compile time.")
    print("     Run 'cargo build' only after this file exists.")

    return {
        "tflite_path": str(tflite_path),
        "quant_params_path": quant_params_path,
        "size_kb": size_kb,
    }


def run_training(
    config_name: str,
    config_path: str = "config/presets/tinyagri.yaml",
    output_dir: str = "results/",
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one named experiment into ``<output_dir>/<config_name>/``.

    Writes ``results.json`` there - the history collapsed to the best epoch -
    and returns what `train_model` returned plus ``results_path``.

    config_overrides: merged on top of the YAML config before training begins.
        Use this to inject dynamic values (e.g. teacher_weights resolved at runtime).
    """
    run_dir = Path(output_dir) / config_name
    run_dir.mkdir(parents=True, exist_ok=True)

    result = train_model(
        config_path=config_path,
        experiment_name=config_name,
        output_dir=str(run_dir),
        extra_config=config_overrides,
    )

    results_path = run_dir / "results.json"
    summary = dict(result, final_metrics=_select_best_epoch_metrics(result["final_metrics"]))
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    return {**result, "results_path": str(results_path)}


def train_and_score(
    model: str,
    config_path: str,
    output_dir: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one model, export it to INT8 unless it is the teacher, and score each format.

    One entry of `run_pipeline`'s summary, and what the tables in
    ``experiments/`` are built from. Each format is scored from its own file
    on the test split, so the quantization gap is measured rather than assumed.

    Args:
        model: The preset's model block to train.
        config_path: Preset the model is trained, calibrated and scored with.
        output_dir: The run goes to ``<output_dir>/<model>/``.
        overrides: Keys applied on top of the preset, as in `run_training`.
            Set ``models_dir`` to keep the weights out of the preset's own
            models directory.

    Returns:
        ``model_path``, ``run_dir``, ``qat_applied``, ``tflite_path``,
        ``tflite_size_kb``, ``quant_params_path``, ``eval_fp32`` and
        ``eval_int8``. The INT8 fields are None for the teacher.
    """
    from nanou.evaluate import evaluate

    res = run_training(model, config_path, output_dir, overrides)
    entry: dict[str, Any] = {
        "model_path": res["model_path"],
        "run_dir": res["experiment_dir"],
        "qat_applied": res["qat_applied"],
        "tflite_path": None,
        "tflite_size_kb": None,
        "quant_params_path": None,
        "eval_fp32": None,
        "eval_int8": None,
    }

    if model != TEACHER:
        print(f"\nQuantizing {model} to INT8 TFLite")
        keras_path = Path(entry["model_path"])
        exported = export_int8(keras_path, models_dir=keras_path.parent, config_path=config_path)
        entry["tflite_path"] = exported["tflite_path"]
        entry["tflite_size_kb"] = exported["size_kb"]
        entry["quant_params_path"] = exported["quant_params_path"]

    print(f"\nEvaluating {model} FP32 Keras on the test split...")
    entry["eval_fp32"] = evaluate(entry["model_path"], config_path, out_dir=entry["run_dir"])
    if entry["tflite_path"]:
        print(f"\nEvaluating {model} INT8 TFLite on the test split...")
        entry["eval_int8"] = evaluate(entry["tflite_path"], config_path, out_dir=entry["run_dir"])
    return entry


def _banner(title: str) -> None:
    print(f"\n{'=' * 55}\n{title}\n{'=' * 55}")


def run_pipeline(
    config_path: str,
    output_dir: str,
    *,
    distil: bool = True,
    model: str | None = None,
) -> dict[str, Any]:
    """Train, export and evaluate, then write ``<output_dir>/summary.json``.

    With ``distil`` (the default) this is QAD: the teacher is trained first and
    the student distils from it, with QAT from epoch 1. Without it, ``model`` is
    trained on its own - the teacher as a standalone baseline, or a student
    without KD as an ablation.

    Every trained model is then scored on the test split as FP32; every model
    except the teacher is also exported to INT8 and scored again in that
    format, so the quantization gap is measured rather than assumed.

    Returns:
        The summary written to disk: ``config_path``, ``distil`` and a ``models``
        dict with one entry per trained model, keyed by model name. The schema
        is the same for both paths; a single-model run just has one entry.

    Raises:
        ValueError: if ``model`` is given with ``distil``, or missing without it.
    """
    if distil and model is not None:
        raise ValueError("model applies only without distil; QAD trains both models")
    if not distil and model is None:
        raise ValueError("without distil, name the model to train (e.g. bu_net)")

    models: dict[str, dict[str, Any]] = {}
    if distil:
        _banner("TEACHER -> STUDENT DISTILLATION PIPELINE")
        print("\nPhase 1: Teacher")
        models[TEACHER] = train_and_score(TEACHER, config_path, output_dir)
        print(f"Teacher trained  ->  {models[TEACHER]['model_path']}")

        print("\nPhase 2: Student")
        # Inject the teacher path resolved at runtime so the student always distils
        # from the model trained in Phase 1, regardless of what is in the YAML.
        models[STUDENT] = train_and_score(
            STUDENT, config_path, output_dir, {"teacher_weights": models[TEACHER]["model_path"]}
        )
        print(f"Student trained  ->  {models[STUDENT]['model_path']}")
    else:
        _banner(f"STANDARD TRAINING: {model.upper()} (No KD)")
        models[model] = train_and_score(model, config_path, output_dir, {"use_distillation": False})
        print(f"Training complete  ->  {models[model]['model_path']}")

    summary = {"config_path": config_path, "distil": distil, "models": models}
    summary_path = Path(output_dir) / SUMMARY_FILE
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\nSummary saved to {summary_path}")
    return summary
