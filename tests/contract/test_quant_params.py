"""Section 8.1: ``nanou_quant_params.json`` must not change, byte for byte.

``firmware/build.rs`` reads this file at compile time and bakes the scales,
zero-points and normalisation constants into the binary as ``f32::from_bits``
literals. A changed value does not fail a build - it silently changes what the
device does to every pixel before inference. This is the hardest constraint in
the refactor, so it is checked three ways: against the captured baseline, for
internal shape, and against the exported graph it claims to describe.
"""

import json

import pytest

from tests.contract import oracle

pytestmark = pytest.mark.contract


def test_bytes_match_baseline(dataset, artifacts, baseline_dir, record):
    """The file on disk is identical to the captured one."""
    live = oracle.sha256_file(artifacts["quant_params"])
    expected = record["artifacts"]["quant_params"]["sha256"]
    assert live == expected, (
        f"{dataset}: quant params changed. The firmware bakes these constants in "
        f"at compile time, so this changes on-device preprocessing silently. "
        f"Diff against {baseline_dir}/nanou_quant_params.json."
    )


def test_structure_is_what_build_rs_reads(artifacts):
    """The keys ``build.rs`` indexes, with the types it expects."""
    params = oracle.load_json(artifacts["quant_params"])

    assert set(params) == {"input", "output", "normalization"}
    for side in ("input", "output"):
        block = params[side]
        assert set(block) == {"dtype", "scale", "zero_point", "shape"}
        assert block["dtype"] == "int8"
        assert isinstance(block["scale"], float) and block["scale"] > 0.0
        assert isinstance(block["zero_point"], int)
        assert len(block["shape"]) == 4 and block["shape"][0] == 1

    norm = params["normalization"]
    assert set(norm) == {"mean", "std"}
    assert len(norm["mean"]) == 3 and len(norm["std"]) == 3
    assert all(s > 0.0 for s in norm["std"]), "a zero std would divide by zero on device"


def test_agrees_with_the_exported_graph(dataset, artifacts):
    """The JSON describes the .tflite sitting next to it, not an older export.

    These two files are written by the same export step but live separately, so
    nothing structural keeps them in sync. If they drift, the host scores one
    graph and the device preprocesses for another.
    """
    params = oracle.load_json(artifacts["quant_params"])
    io = oracle.tflite_io(artifacts["tflite"])

    for side in ("input", "output"):
        assert params[side]["dtype"] == io[side]["dtype"], f"{dataset}: {side} dtype"
        assert params[side]["shape"] == io[side]["shape"], f"{dataset}: {side} shape"
        assert params[side]["scale"] == pytest.approx(io[side]["scale"], rel=0, abs=0), (
            f"{dataset}: {side} scale differs from the graph it describes"
        )
        assert params[side]["zero_point"] == io[side]["zero_point"], f"{dataset}: {side} zero point"


def test_normalization_matches_the_preset(dataset, artifacts):
    """The constants baked into the firmware are the ones training used."""
    from nanou.config import resolve

    params = oracle.load_json(artifacts["quant_params"])
    norm = resolve(oracle.DATASETS[dataset]).normalization

    assert params["normalization"]["mean"] == pytest.approx(norm["mean"])
    assert params["normalization"]["std"] == pytest.approx(norm["std"])


def test_serializes_stably(artifacts, tmp_path):
    """Re-writing the parsed JSON reproduces the file, so a round trip is safe."""
    with open(artifacts["quant_params"]) as fh:
        original = fh.read()
    rewritten = json.dumps(json.loads(original), indent=2)
    assert rewritten.strip() == original.strip()
