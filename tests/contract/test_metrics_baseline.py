"""Segmentation metrics on the held-out split must not move.

The models and the split are fixed, so this is not a statistical comparison -
the same weights over the same 181 frames should give the same numbers. The
tolerance exists only for float reduction order, which is why the INT8 path
(integer arithmetic, deterministic) is held tighter than the float one.
"""

import os

import pytest

from tests.contract import oracle

pytestmark = pytest.mark.contract

# INT8 inference is bit-exact; anything above this is a real change.
INT8_TOL = 1e-9
# The float path goes through XNNPACK/oneDNN, whose reduction order is not pinned,
# and it is not pinned across devices either: the same .h5 over the same split
# scores mIoU 0.88687270 on CPU and 0.88680550 on GPU, a 6.7e-5 gap. The
# baseline is captured on whichever device is free, so the tolerance has to
# clear that. Real evaluation drift is orders of magnitude larger.
FP32_TOL = 1e-3
# The README's headline claim: quantisation is free. 1 mIoU point is the bound
# the README's "no accuracy loss" wording has to clear.
QUANT_GAP = 0.01


@pytest.mark.parametrize("fmt,tol", [("int8", INT8_TOL), ("fp32", FP32_TOL)])
def test_scores_match_baseline(dataset, record, scored, fmt, tol):
    if fmt not in record.get("metrics", {}):
        pytest.skip(f"{dataset}: no {fmt} metrics in the baseline")
    live = oracle.flat_metrics(scored(dataset, fmt))
    expected = record["metrics"][fmt]

    drifted = {k: (expected[k], live[k]) for k in expected if abs(live[k] - expected[k]) > tol}
    assert not drifted, f"{dataset} {fmt}: metrics moved (baseline, now): {drifted}"


def test_quantization_costs_nearly_nothing(dataset, scored):
    """FP32 and INT8 agree - the claim the whole QAD pipeline exists to make."""
    fp32 = scored(dataset, "fp32")
    int8 = scored(dataset, "int8")
    gap = abs(fp32["miou"] - int8["miou"])
    assert gap < QUANT_GAP, (
        f"{dataset}: quantisation moved mIoU by {gap:.4f} "
        f"(fp32 {fp32['miou']:.4f} vs int8 {int8['miou']:.4f})"
    )


def test_scored_the_whole_split(dataset, scored, record):
    """A silently shortened split would make every metric above meaningless."""
    n = scored(dataset, "int8")["n_images"]
    assert n > 0
    summary = os.path.join(oracle.project_root(), "baseline", dataset, "metrics_int8.json")
    if os.path.exists(summary):
        assert n == oracle.load_json(summary)["n_images"]


def test_matches_the_shipped_pipeline_summary(dataset, artifacts, scored):
    """Cross-check against the run that produced the shipped models.

    ``results/<dataset>/pipeline_summary.json`` was written by the original
    pipeline. Reproducing its numbers with today's code proves the evaluation
    path has not drifted since the models were trained - a second, independent
    oracle that predates ``baseline/``.
    """
    path = artifacts["summary"]
    if not os.path.exists(path):
        pytest.skip(f"{dataset}: no shipped pipeline summary")
    summary = oracle.load_json(path)
    student = summary.get("student", {})

    # int8 only. The shipped nanou.h5 was overwritten by a pipeline run on
    # 2026-09-08 and no copy of the original survives, so the fp32 half of this
    # summary describes a model that is no longer on disk. The shipped tflite is
    # still the one this summary was written for, and the fp32 path keeps its
    # oracle in ``baseline/``, re-captured against the replacement model.
    checked = 0
    for fmt in ("int8",):
        expected = student.get(f"eval_{fmt}")
        if not expected:
            continue
        live = oracle.flat_metrics(scored(dataset, fmt))
        for key, value in oracle.flat_metrics(expected).items():
            assert live[key] == pytest.approx(value, abs=1e-6), (
                f"{dataset} {fmt}: {key} differs from the shipped summary"
            )
            checked += 1
    if not checked:
        pytest.skip(f"{dataset}: summary carries no student metrics")
