"""The three parameter counts the README quotes.

4,212 float32 -> 4,688 once QAT wraps the graph in fake-quant nodes -> 3,357
stored constants after batch-norm folds into the preceding convolutions at
export. These are literals on purpose: they are published numbers, so the test
fails if the architecture drifts even when the baseline was recaptured with the
drift already in it.
"""

import os

import pytest

from tests.contract import oracle

pytestmark = pytest.mark.contract


def test_float_architecture_is_4212():
    assert oracle.float_param_count() == oracle.FLOAT_PARAMS


def test_qat_checkpoint_is_4688(dataset, artifacts):
    if not os.path.exists(artifacts["keras"]):
        pytest.skip(f"{dataset}: no saved Keras student")
    assert oracle.keras_param_count(artifacts["keras"]) == oracle.QAT_PARAMS


def test_exported_int8_graph_is_3357(dataset, artifacts):
    counts = oracle.tflite_param_counts(artifacts["tflite"])
    assert counts["total"] == oracle.INT8_PARAMS, f"{dataset}: {counts}"


def test_int8_split_is_weights_plus_biases(dataset, artifacts):
    """3,063 int8 weights and 294 int32 biases - nothing left in float."""
    counts = oracle.tflite_param_counts(artifacts["tflite"])
    assert counts.get("int8") == 3063, f"{dataset}: {counts}"
    assert counts.get("int32") == 294, f"{dataset}: {counts}"
    assert "float32" not in counts, f"{dataset}: float constants survived export: {counts}"


def test_counts_match_baseline(dataset, artifacts, record):
    """Cross-check against the capture, which also pins the QAT count per dataset."""
    assert oracle.float_param_count() == record["param_counts"]["float"]
    assert oracle.tflite_param_counts(artifacts["tflite"]) == record["param_counts"]["int8"]
    if record["param_counts"]["qat"] is not None and os.path.exists(artifacts["keras"]):
        assert oracle.keras_param_count(artifacts["keras"]) == record["param_counts"]["qat"]
