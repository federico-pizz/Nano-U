# NAS utilities and pipeline usage

This document explains how the project's NAS utilities integrate into the Python
TF pipeline (`scripts/tf_pipeline.py`) and gives concrete command examples.

Overview
- `src/nas_covariance.py`: core utilities (ActivationExtractor, RunningCovariance,
  covariance_redundancy, FeatureDecorrelationRegularizer, DistillationAwareNAS).
- `src/nas_analyze.py`: minimal analysis runner that computes redundancy scores
  for a list of teacher layer names and writes JSON results.
- `src/train_nas.py`: NAS-enabled training entrypoint that adds covariance-based
  regularization into the training loop when requested.
- `scripts/tf_pipeline.py`: orchestration CLI that wraps build/train/distill/analyze/quantize/infer/eval steps.

Pipeline usage

Activate the TF virtual environment before running the pipeline:

  source .venv-tf/bin/activate

Run the pipeline commands via the Python CLI (no shell script required):

- Build models:

  python scripts/tf_pipeline.py build

- Train a model (standard training):

  python scripts/tf_pipeline.py train --model bu_net

- Train with NAS-enabled regularization (uses `src/train_nas.py`):

  python scripts/tf_pipeline.py train --model nano_u --enable-nas --nas-layers "/conv/" --nas-weight 0.01

  Notes:
  - `--enable-nas` will cause the pipeline to invoke `src/train_nas.py` instead
    of the default `src/train.py`.
  - `--nas-layers` accepts comma-separated layer selectors (exact names or
    simple regex-wrapped selectors, e.g. `/conv_/`). If omitted, train_nas
    defaults to a conservative selector set.

- Distillation training (teacher -> student):

  python scripts/tf_pipeline.py distill --model nano_u --teacher-weights models/bu_net.keras

- Analyze layer redundancy with the minimal analyzer:

  python scripts/tf_pipeline.py analyze --num-batches 50

  The pipeline will invoke `src/nas_analyze.py` which writes JSON scores to the path
  you pass via its --out argument when used directly.

- Quantize a model (delegates to src/quantize.py):

  python scripts/tf_pipeline.py quantize --model-name bu_net --output models/bu_net.tflite

- Run inference on a model (delegates to src/infer.py):

  python scripts/tf_pipeline.py infer models/bu_net.tflite

- Evaluate a model (delegates to src/evaluate.py):

  python scripts/tf_pipeline.py eval --model-name bu_net --out results/eval/bu_net_eval.png

Implementation notes

- The pipeline is intentionally lightweight and delegates heavy work to existing
  modules in `src/` to avoid duplicating logic. The NAS-enabled trainer is
  implemented as `src/train_nas.py` so the main `src/train.py` remains untouched.

- `src/nas_analyze.py` is a minimal runner: for production analysis, replace the
  synthetic dataset in the script with your validation dataset (or call the
  script with a wrapper that constructs a real tf.data dataset).

- Logging: scripts use Python logging. Replace or extend with the project's
  logging facility in `src/utils/config.py` if required.

Testing

- Unit tests for NAS utilities live in `tests/test_nas_covariance.py`.
- To exercise the pipeline in dry-run mode, run the commands above after
  activating `.venv-tf`.

Troubleshooting

- If `scripts/tf_pipeline.py` complains about a missing virtualenv, create one:

  python3 -m venv .venv-tf
  source .venv-tf/bin/activate
  python -m pip install -r requirements.txt

- If `src/train_nas.py` fails because helper functions are not exported from
  `src/train.py`, implement small adapter helpers in `src/train.py` (e.g.
  build_model, get_dataset, compile_and_train) or call `src/train.py` directly
  with subprocess in the pipeline.
