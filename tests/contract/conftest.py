"""Fixtures for the contract suite.

Every test here compares the repository against ``baseline/``, which is not
committed. When it is absent the tests skip with the command that creates it
rather than failing - an oracle you have not captured yet is not a regression.
"""

import os

import pytest

from tests.contract import oracle

CAPTURE_CMD = "python -m tests.contract.capture_baseline"


def pytest_generate_tests(metafunc):
    """Run every test that takes a ``dataset`` argument once per dataset."""
    if "dataset" in metafunc.fixturenames:
        metafunc.parametrize("dataset", sorted(oracle.DATASETS))


@pytest.fixture(scope="session")
def baseline_root():
    root = os.path.join(oracle.project_root(), "baseline")
    if not os.path.isdir(root):
        pytest.skip(f"no baseline captured. Capture it with:  {CAPTURE_CMD}")
    return root


@pytest.fixture
def record(baseline_root, dataset):
    """The captured measurements for one dataset."""
    path = os.path.join(baseline_root, dataset, "record.json")
    if not os.path.exists(path):
        pytest.skip(f"no baseline for {dataset}. Capture it with:  {CAPTURE_CMD}")
    return oracle.load_json(path)


@pytest.fixture
def baseline_dir(baseline_root, dataset):
    return os.path.join(baseline_root, dataset)


@pytest.fixture
def artifacts(dataset):
    """Shipped artefact paths, skipping the dataset when the model is absent.

    The quant-params JSON is not committed, so it is regenerated from the tracked
    .tflite when missing or older than it, exactly as scripts/flash.py does.
    """
    from nanou.config import preset_path
    from nanou.device.flash import regenerate_quant_params
    from nanou.preflight import quant_params_are_stale

    paths = oracle.artifact_paths(dataset)
    if not os.path.exists(paths["tflite"]):
        pytest.skip(f"{dataset}: no exported model at {paths['tflite']}")
    models_dir = os.path.dirname(paths["tflite"])
    if quant_params_are_stale(models_dir, oracle.MODEL_NAME):
        regenerate_quant_params(
            paths["tflite"], models_dir, oracle.MODEL_NAME, preset_path(oracle.DATASETS[dataset])
        )
    return paths


@pytest.fixture(scope="session")
def scored():
    """Cache of ``(dataset, format) -> metrics``, computed at most once each."""
    cache = {}

    def score(dataset, fmt):
        key = (dataset, fmt)
        if key not in cache:
            paths = oracle.artifact_paths(dataset)
            model = paths["tflite"] if fmt == "int8" else paths["keras"]
            if not os.path.exists(model):
                pytest.skip(f"{dataset}: no {fmt} model on disk")
            split_dir = os.path.join(oracle.project_root(), "data", dataset, "test")
            if not os.path.isdir(split_dir):
                pytest.skip(
                    f"{dataset}: no test split. Fix it with:  "
                    f"python scripts/get_data.py --preset {oracle.DATASETS[dataset]}"
                )
            cache[key] = oracle.score_split(oracle.DATASETS[dataset], model, split="test")
        return cache[key]

    return score
