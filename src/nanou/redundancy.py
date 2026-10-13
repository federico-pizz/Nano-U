"""How redundant a layer's channel space is, measured per epoch.

Observation only. Nothing here touches the loss, the gradients, the learning
rate or checkpoint selection - turning `track_redundancy` on or off cannot
change the weights a run produces, only what it writes down. No architecture is
searched and none is pruned: the SVD condition number of the centred
activations is logged so that width decisions can be made by hand from the
evidence. The CSV column names (`<layer>_redundancy_score`,
`_condition_number`, `_rank`, `_num_channels`) are unchanged, so existing
`metrics.csv` files stay readable.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras


def compute_layer_redundancy(activations: tf.Tensor, eps: float = 1e-6) -> dict[str, float]:
    """Compute stable redundancy score using SVD decomposition.

    Args:
        activations: Layer activations tensor (batch, height, width, channels)
        eps: Small value to prevent division by zero

    Returns:
        Dictionary with redundancy metrics
    """
    # Reshape to (samples, features) for SVD
    ndim = len(activations.shape)
    if ndim == 2:
        reshaped = activations
        channels = activations.shape[-1]
    else:
        batch_size, height, width, channels = activations.shape
        reshaped = tf.reshape(activations, (-1, channels))

    # Center activations
    mean_act = tf.reduce_mean(reshaped, axis=0)
    centered = reshaped - mean_act

    # SVD for numerical stability
    s, u, v = tf.linalg.svd(centered, full_matrices=False)
    # Clamp to avoid zeros
    singular_values = tf.maximum(s, eps)

    # Condition number using SVD
    condition_number = tf.reduce_max(singular_values) / tf.reduce_min(singular_values)

    # Redundancy score (normalized): 0 means no redundancy, 1 means highly redundant
    redundancy = 1.0 - (1.0 / (1.0 + tf.math.log(condition_number)))

    def _numpy(x):
        return int(x.numpy()) if hasattr(x, "numpy") else int(x)

    return {
        "redundancy_score": float(redundancy.numpy()),
        "condition_number": float(condition_number.numpy()),
        "rank": _numpy(
            tf.reduce_sum(
                tf.cast(singular_values > tf.reduce_max(singular_values) * 1e-4, tf.int32)
            )
        ),
        "num_channels": _numpy(channels),
    }


def extract_activations(
    model: keras.Model, layer_names: list[str], x: tf.Tensor
) -> dict[str, tf.Tensor]:
    """Extract activations from specified layers.

    Args:
        model: Keras model
        layer_names: List of layer names to extract
        x: Input batch

    Returns:
        Dictionary of layer name to activation tensor
    """
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    intermediate_model = keras.Model(inputs=model.input, outputs=layer_outputs)

    # Inference mode: BatchNorm reads its moving statistics and nothing updates,
    # so measuring a layer cannot change the model being measured.
    activations = intermediate_model(x, training=False)
    # A single-output model hands back a bare tensor, not a one-element list,
    # and zipping over a bare tensor would walk its batch dimension instead.
    if not isinstance(activations, (list, tuple)):
        activations = [activations]

    return dict(zip(layer_names, activations, strict=True))


def compute_redundancy_metrics(
    model: keras.Model, x: tf.Tensor, layers_to_monitor: list[str]
) -> dict[str, dict[str, float]]:
    """Compute redundancy metrics for all specified layers.

    Args:
        model: Keras model
        x: Input batch
        layers_to_monitor: List of layer names to monitor

    Returns:
        Dictionary of layer name to redundancy metrics
    """
    # Extract activations
    activations_dict = extract_activations(model, layers_to_monitor, x)

    # Compute metrics for each layer
    metrics = {}
    for layer_name, activations in activations_dict.items():
        metrics[layer_name] = compute_layer_redundancy(activations)

    return metrics


class RedundancyMonitor(keras.callbacks.Callback):
    """Log per-layer SVD redundancy metrics every `log_frequency` epochs (or batches)."""

    def __init__(
        self,
        layers_to_monitor: list[str] = None,
        log_frequency: int = 1,
        output_dir: str = "redundancy_logs/",
        validation_data: tf.data.Dataset | tuple[np.ndarray, np.ndarray] | None = None,
        monitor_frequency: str = "epoch",
        log_dir: str = None,
        **kwargs,
    ):
        super().__init__()
        self.layers_to_monitor = layers_to_monitor or ["conv2d", "conv2d_1"]
        self.log_frequency = log_frequency
        self.output_dir = str(output_dir)
        self.monitor_frequency = monitor_frequency  # 'epoch' or 'batch'
        self.validation_data = validation_data
        self.log_dir = log_dir

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.log_dir:
            self.writer = tf.summary.create_file_writer(self.log_dir)

        self.redundancy_history: list[dict[str, Any]] = []
        self.metrics: defaultdict[str, list[Any]] = defaultdict(list)
        self.batch_count = 0

    def _get_batch(self):
        """Get a batch for metrics (validation data or training data)."""
        val = None
        if self.validation_data is not None:
            val = self.validation_data
        elif hasattr(self.model, "validation_data") and self.model.validation_data is not None:
            val = self.model.validation_data

        if val is not None:
            if isinstance(val, tf.data.Dataset):
                for x, _ in val.take(1):
                    return x[: min(16, tf.shape(x)[0])]
            elif isinstance(val, (list, tuple)) and len(val) > 0:
                x = val[0]
                return x[: min(16, len(x))]

        # Fallback to test batch if set by unit tests
        if hasattr(self, "_test_x_batch"):
            return self._test_x_batch
        return None

    def on_batch_end(self, batch: int, logs: dict[str, float] | None = None):
        """Log redundancy every `log_frequency` batches, in batch mode."""
        self.batch_count += 1
        if self.monitor_frequency == "batch" and self.batch_count % self.log_frequency == 0:
            self._compute_and_log(f"step_{self.batch_count}")

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None):
        """Log redundancy every `log_frequency` epochs, in epoch mode."""
        if self.monitor_frequency == "epoch" and (epoch + 1) % self.log_frequency == 0:
            self._compute_and_log(epoch)

    def _compute_and_log(self, identifier: int | str):
        x_batch = self._get_batch()
        if x_batch is None:
            return

        row: dict[str, int | str | float] = {
            "epoch" if isinstance(identifier, int) else "step": identifier
        }
        for name in self.layers_to_monitor:
            try:
                # 1. Try exact match
                # 2. Try 'quant_' prefix (TFMOT default)
                # 3. Try finding any layer that contains the name
                layer = None
                for candidate in [name, f"quant_{name}"]:
                    try:
                        layer = self.model.get_layer(candidate)
                        break
                    except (ValueError, KeyError):
                        continue

                if layer is None:
                    # Final attempt: partial match
                    for candidate in self.model.layers:
                        if name in candidate.name:
                            layer = candidate
                            break

                if layer is None:
                    raise KeyError(f"Layer '{name}' (or 'quant_{name}') not found")

                act = extract_activations(self.model, [layer.name], x_batch)[layer.name]
                r = compute_layer_redundancy(act)
                row[f"{name}_redundancy_score"] = r["redundancy_score"]
                row[f"{name}_condition_number"] = r["condition_number"]
                row[f"{name}_rank"] = float(r["rank"])
                row[f"{name}_num_channels"] = float(r["num_channels"])
            except Exception as e:
                if not hasattr(self, "_skipped_layers"):
                    self._skipped_layers = set()
                if name not in self._skipped_layers:
                    print(f"Redundancy monitor skipping layer {name}: {e}")
                    self._skipped_layers.add(name)
                continue

        self.redundancy_history.append(row)
        for k, v in row.items():
            if k not in ["epoch", "step"]:
                self.metrics[k].append(v)

        if self.log_dir:
            with self.writer.as_default():
                step = identifier if isinstance(identifier, int) else self.batch_count
                for k, v in row.items():
                    if k not in ["epoch", "step"]:
                        tf.summary.scalar(f"redundancy/{k}", v, step=step)
            self.writer.flush()

        csv_path = Path(self.output_dir) / "metrics.csv"
        pd.DataFrame(self.redundancy_history).to_csv(csv_path, index=False)

    def on_train_end(self, logs: dict[str, float] | None = None):
        """Write the collected history to metrics.csv one last time."""
        if self.redundancy_history:
            csv_path = Path(self.output_dir) / "metrics.csv"
            pd.DataFrame(self.redundancy_history).to_csv(csv_path, index=False)

    def get_metrics(self) -> dict[str, list[float]]:
        """Return the per-layer metric series collected so far."""
        return dict(self.metrics)
