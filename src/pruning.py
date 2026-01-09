"""
Lightweight magnitude pruning for an already quantized INT8 TFLite model.

The script rewrites the FlatBuffer in-place using the TensorFlow Lite schema,
zeroing the smallest-magnitude INT8 weights to reach a requested sparsity
target. This does not require access to the original training graph and keeps
the int8 quantization parameters untouched.

Usage example:
	python src/pruning.py --input models/Nano_U_int8.tflite \
        --output models/Nano_U_int8_pruned.tflite --sparsity 0.7
Notes:
- Only INT8 constant buffers are pruned. Activations and non-INT8 tensors are
  left unchanged.
- After pruning you can optionally re-run post-training quantization for
  further optimization, but it is not required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb


TensorRef = Tuple[int, int, schema_fb.TensorT, schema_fb.BufferT]


def _load_mutable_model(model_path: Path) -> schema_fb.ModelT:
	"""Load a TFLite flatbuffer into a mutable ModelT object."""
	raw_buffer = bytearray(model_path.read_bytes())
	model_obj = schema_fb.Model.GetRootAsModel(raw_buffer, 0)
	return schema_fb.ModelT.InitFromObj(model_obj)


def _iter_int8_weight_tensors(model: schema_fb.ModelT) -> Iterable[TensorRef]:
	"""Yield INT8 tensors backed by constant buffers (likely weights)."""
	for sg_idx, subgraph in enumerate(model.subgraphs):
		input_buffers = set(subgraph.inputs)
		for t_idx, tensor in enumerate(subgraph.tensors):
			buf_idx = tensor.buffer
			buffer = model.buffers[buf_idx]
			if buffer.data is None or len(buffer.data) == 0:
				continue
			if buf_idx in input_buffers:
				continue  # skip model inputs
			if tensor.type != schema_fb.TensorType.INT8:
				continue
			yield sg_idx, t_idx, tensor, buffer


def _compute_threshold(weights: np.ndarray, sparsity: float, min_abs: float | None) -> float:
	"""Compute magnitude threshold so that ~sparsity fraction becomes zero."""
	sparsity = min(max(sparsity, 0.0), 0.99)
	if sparsity == 0.0:
		return float("inf")
	k_zero = int(weights.size * sparsity)
	if k_zero <= 0:
		return float("inf")
	abs_w = np.abs(weights)
	threshold = np.partition(abs_w, k_zero)[k_zero]
	if min_abs is not None:
		threshold = max(threshold, float(min_abs))
	return threshold


def prune_int8_tflite(input_path: Path, output_path: Path, sparsity: float, min_abs: float | None) -> None:
	"""Magnitude-prune INT8 weight buffers and save a new TFLite file."""
	model = _load_mutable_model(input_path)
	total_params = 0
	zeroed_params = 0

	for sg_idx, t_idx, tensor, buffer in _iter_int8_weight_tensors(model):
		weights = np.frombuffer(buffer.data, dtype=np.int8)
		threshold = _compute_threshold(weights, sparsity, min_abs)
		if not np.isfinite(threshold):
			continue
		mask = np.abs(weights) <= threshold
		before_nnz = np.count_nonzero(weights)
		weights = weights.copy()
		weights[mask] = 0
		buffer.data = bytearray(weights.tobytes())

		total_params += weights.size
		zeroed_params += int(mask.sum())

		print(
			f"Pruned subgraph {sg_idx} tensor {t_idx}: "
			f"{before_nnz}->{np.count_nonzero(weights)} non-zeros (threshold={threshold:.4f})"
		)

	builder = flatbuffers.Builder(1024 * 1024)
	builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
	output_path.write_bytes(builder.Output())

	if total_params == 0:
		print("No INT8 constant tensors were pruned. Did you pass the right model?")
	else:
		achieved = zeroed_params / total_params
		print(
			f"Saved pruned model to {output_path} | zeroed {zeroed_params}/{total_params} "
			f"params (~{achieved:.2%} sparsity)"
		)


def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Prune an INT8 TFLite model by magnitude.")
	parser.add_argument("--input", type=Path, default=Path("models/Nano_U_int8.tflite"), help="Path to the INT8 TFLite model")
	parser.add_argument("--output", type=Path, default=Path("models/Nano_U_int8_pruned.tflite"), help="Where to write the pruned model")
	parser.add_argument("--sparsity", type=float, default=0.7, help="Fraction of smallest-magnitude weights to zero (0-0.99)")
	parser.add_argument("--min-abs", type=float, default=None, help="Optional absolute threshold floor to keep very small weights zeroed")
	return parser.parse_args(argv)


def main(argv: list[str]) -> None:
	args = parse_args(argv)
	prune_int8_tflite(args.input, args.output, args.sparsity, args.min_abs)


if __name__ == "__main__":
	main(sys.argv[1:])
